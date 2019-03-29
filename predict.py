import os
import base64
import io
import googleapiclient.discovery
import time
import pickle
from PIL import Image

def predict_online(project, model, version, instances):
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}/versions/{}'.format(project, model,version)

    response = service.projects().predict(name=name,body={'instances': instances}).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']

def main():
    project = 'PROJECT_NAME' 
    model = 'MODEL_NAME'
    version = 'VERSION_NUMBER'

    height = width = 224

    PATH_TO_EVAL_DIR = 'images/test_data_images'
    EVAL_IMAGES = [os.path.join(PATH_TO_EVAL_DIR, img_name) for img_name in os.listdir(PATH_TO_EVAL_DIR)]
    PATH_TO_PREDICTION_DIR = 'resnet101/prediction'

    json_inp = []
    for i, img_path in enumerate(EVAL_IMAGES):
        img = Image.open(img_path)
        output_str = io.BytesIO()
        img.save(output_str, "JPEG")
        json_inp.append({"inputs":{"b64": base64.b64encode(output_str.getvalue()).decode('ascii')}})
        output_str.close()
        
    results = []
    batch_size = 3
    first_time = time.time()
    
    for i in range(int(len(EVAL_IMAGES)/batch_size)):
        start_index = i * batch_size
        start_time = time.time()
        instance = json_inp[start_index:start_index + batch_size]
        status = True
        while True:
            try:
                result = predict_online(project, model, instance, version)
            except Exception as e:
                print(str(e))
                continue
            break

        results.append(result)
        print('Batch: {}\tElapsed time: {}'.format(i, time.time() - start_time))

        if i % 5 == 0:
            with open('frcnn_resnet101_results.pickle', 'wb') as f:
                print('Saving prediction')
                pickle.dump(results,f)

    print('-------------\nTotal time consumed: {}'.format(time.time() - first_time))
    
    # Write to file
    i = 0
    for result_set in results:
        for result in result_set:
            img_name = EVAL_IMAGES[i]
            dest_file_name = str.split(img_name, '.')[0] + '.txt'
            num_detections = int(result['num_detections'])
            pred_classes = result['detection_classes'][:num_detections]
            pred_confidence = result['detection_scores'][:num_detections]
            pred_bnbboxes = result['detection_boxes'][:num_detections]

            with open(os.path.join(PATH_TO_PREDICTION_DIR,dest_file_name), 'w') as out_file:
                for j in range(num_detections):
                    line = str(pred_classes[j]) + ' ' 
                    line += str(pred_confidence[j]) + ' '

                    line += str(pred_bnbboxes[j][0] * 224) + ' ' # xmin
                    line += str(pred_bnbboxes[j][1] * 224) + ' ' # ymin
                    line += str(pred_bnbboxes[j][2] * 224) + ' ' # xmax
                    line += str(pred_bnbboxes[j][3] * 224) + '\n' # ymax


                    out_file.write(line)
            i += 1
            out_file.close()
    print('Prediction complete')
