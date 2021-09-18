import time
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from PIL import Image
from evaluate_fid import Evaluation
from preprocessing import DataPipeline
from flask import Flask, flash, request, redirect, render_template
# from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
# datahasil = os.listdir('static/result/')

print("Init Flask App")
# app = Flask(__name__, static_url_path='/data_science_product/static')
app = Flask(__name__)


@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


""" Edit Start """
# Error Handle


@app.route("/")
# def index():
#     # https://riset.informatika.umm.ac.id/area_of_interest/{aoi_id}
#     return redirect('https://riset.informatika.umm.ac.id/area_of_interest/1')
# Compare Model f_nim Brain Tumor Disease
@app.route('/1/compare')
def f_201710370311030_compare():
    return render_template('/201710370311030/compare.html', )


@app.route('/1/pred_comp', methods=['POST'])
def f_201710370311030_predict_compare():  # Ganti nim dengan NIM anda misal : _2132131312
    # initialitation variable
    respon_model = []
    running_time = []
    dir_result_img = []
    fid_local = []
    fid_global = []
    chosen_model = request.form.getlist('select_model')
    is_evaluate = request.form['is_evaluate_fid']
    list_image_patch = request.form.getlist('select_image_patch')
    model_dict = {
        'GAN_C1': 'static/model/201710370311030/GAN_C1.h5',
        'GAN_C10': 'static/model/201710370311030/GAN_C10.h5',
        'GAN_C100': 'static/model/201710370311030/GAN_C100.h5',
        'GAN_AUG': 'static/model/201710370311030/GAN_AUG.h5',
        'GAN_NO_AUG': 'static/model/201710370311030/GAN_NO_AUG.h5'
    }

    # Load patch batik from directory
    patchA = image.load_img(list_image_patch[0], target_size=(32, 32))
    patchB = image.load_img(list_image_patch[1], target_size=(32, 32))

    # convert patch batik to array
    patchA = image.img_to_array(patchA)
    patchB = image.img_to_array(patchB)

    # normalization array patch batik
    patchA = (patchA-127.5)/127.5
    patchA = np.expand_dims(patchA, axis=0)
    patchB = (patchB-127.5)/127.5
    patchB = np.expand_dims(patchB, axis=0)

    # load dataset
    dataset = DataPipeline().execute()

    # img = cv2.cvtColor(
    #     np.array(np.array(Image.open(filename))), cv2.COLOR_BGR2RGB
    # )

    # Isi dengan Nama model dan path lokasi model disimpan (Pastikan menyimpan model dalam folder /static/model/nim/namamodel.h5)
    # Jika pakai Json dan weight model saat menyimpan model gunakan kode ini
    # Beri kode nama "_js" tanpa petik di akhir nama model

    for m in chosen_model:
        # if "_js" in m:  # dari kode nama model _js selanjutnya program akan membaca model format json dengan block kode dalam if
        #     json_file = open(model_dict[m][0], 'r')
        #     loaded_model_json = json_file.read()
        #     json_file.close()
        #     model = model_from_json(loaded_model_json)
        #     model.load_weights(model_dict[m][1])
        # else:  # bila nama model tidak mengandung kode nama _js maka model akan di muat menggunakan load_model()
        model = load_model(model_dict[m])

        # preprocessing gambar lakukan sesuai dengan preprocessing yang sama saat proses training
        # imgs = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[
        #                       0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)

        # mulai prediksi
        start = time.time()
        img_generate = model([patchA, patchB])

        # evaluate result batik generate
        if is_evaluate == "yes":
          evaluate = Evaluation(dataset=dataset, model=model)
          _fid_local, _fid_global = evaluate.fid(patchA, patchB, img_generate)
          fid_local.append(round(_fid_local,2))
          fid_global.append(round(_fid_global,2))

        # save result batik generate to directory
        prediction = image.array_to_img(img_generate[0])
        prediction.save(
            "static/result/201710370311030/{}.png".format(m)
        )
        dir_result_img.append("result/201710370311030/{}.png".format(m))
        running_time.append(round(time.time()-start, 4)
                            )  # hitung waktu prediksi
        # respon_model.append([round(elem * 100, 2)
        #                     for elem in pred])  # hitung nilai prediksi

    # return str("fid_global: {} \n fid_local: {}".format(fid_global, fid_local))
    return f_201710370311030_predict_result_compare(respon_model, chosen_model, running_time, list_image_patch, dir_result_img, fid_local, fid_global, "no")


@app.route('/1/pred_comps', methods=['POST'])
def f_201710370311030_predicts_compare():

    respon_model = []
    running_time = []
    dir_result_img = []
    fid_local = []
    fid_global = []
    chosen_model = request.form.getlist('select_model')
    patchA = request.files["patchA"]
    patchA.save(os.path.join('static', 'patchA.jpg'))
    patchB = request.files["patchB"]
    patchB.save(os.path.join('static', 'patchB.jpg'))
    list_image_patch = [
      'patchA.jpg',
      'patchB.jpg'
    ]
    
    model_dict = {
        'GAN_C1': 'static/model/201710370311030/GAN_C1.h5',
        'GAN_C10': 'static/model/201710370311030/GAN_C10.h5',
        'GAN_C100': 'static/model/201710370311030/GAN_C100.h5',
        'GAN_AUG': 'static/model/201710370311030/GAN_AUG.h5',
        'GAN_NO_AUG': 'static/model/201710370311030/GAN_NO_AUG.h5'
    }

    # Load patch batik from directory
    patchA = image.load_img(os.path.join('static', list_image_patch[0]), target_size=(32, 32))
    patchB = image.load_img(os.path.join('static', list_image_patch[1]), target_size=(32, 32))

    # convert patch batik to array
    patchA = image.img_to_array(patchA)
    patchB = image.img_to_array(patchB)

    # normalization array patch batik
    patchA = (patchA-127.5)/127.5
    patchA = np.expand_dims(patchA, axis=0)
    patchB = (patchB-127.5)/127.5
    patchB = np.expand_dims(patchB, axis=0)

    # load dataset
    dataset = DataPipeline().execute()

    # for m in chosen_model:
    #     if "_js" in m:  # dari kode nama model _js selanjutnya program akan membaca model format json dengan block kode dalam if
    #         json_file = open(model_dict[m][0], 'r')
    #         loaded_model_json = json_file.read()
    #         json_file.close()
    #         model = model_from_json(loaded_model_json)
    #         model.load_weights(model_dict[m][1])
    #     else:  # bila nama model tidak mengandung kode nama _js maka model akan di muat menggunakan load_model()
    #         model = load_model(model_dict[m])

    #     # preprocessing gambar lakukan sesuai dengan preprocessing yang sama saat proses training
    #     imgs = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[
    #                           0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)

    #     # mulai prediksi
    #     start = time.time()
    #     pred = model.predict(imgs)[0]
    #     running_time.append(round(time.time()-start, 4)
    #                         )  # hitung waktu prediksi
    #     respon_model.append([round(elem * 100, 2)
    #                         for elem in pred])  # hitung nilai prediksi

    # return f_201710370311030_predict_result_compare(respon_model, chosen_model, running_time, 'temp.jpg')
    for m in chosen_model:
        # if "_js" in m:  # dari kode nama model _js selanjutnya program akan membaca model format json dengan block kode dalam if
        #     json_file = open(model_dict[m][0], 'r')
        #     loaded_model_json = json_file.read()
        #     json_file.close()
        #     model = model_from_json(loaded_model_json)
        #     model.load_weights(model_dict[m][1])
        # else:  # bila nama model tidak mengandung kode nama _js maka model akan di muat menggunakan load_model()
        model = load_model(model_dict[m])

        # preprocessing gambar lakukan sesuai dengan preprocessing yang sama saat proses training
        # imgs = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[
        #                       0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)

        # mulai prediksi
        start = time.time()
        img_generate = model([patchA, patchB])

        # evaluate result batik generate
        # if is_evaluate == "yes":
        #   evaluate = Evaluation(dataset=dataset, model=model)
        #   _fid_local, _fid_global = evaluate.fid(patchA, patchB, img_generate)
        #   fid_local.append(round(_fid_local,2))
        #   fid_global.append(round(_fid_global,2))

        # save result batik generate to directory
        prediction = image.array_to_img(img_generate[0])
        prediction.save(
            "static/result/201710370311030/{}.png".format(m)
        )
        dir_result_img.append("result/201710370311030/{}.png".format(m))
        running_time.append(round(time.time()-start, 4)
                            )  # hitung waktu prediksi
        # respon_model.append([round(elem * 100, 2)
        #                     for elem in pred])  # hitung nilai prediksi

    # return str("fid_global: {} \n fid_local: {}".format(fid_global, fid_local))
    return f_201710370311030_predict_result_compare(respon_model, chosen_model, running_time, list_image_patch, dir_result_img, fid_local, fid_global, "yes")


def f_201710370311030_predict_result_compare(probs, mdl, run_time, img, dir_result_img, fid_local, fid_global, is_patch_input):

    # isi dengan nama kelas 1 sampai ke n sesuai dengan urutan kelas data pada classification report key di isi dengan nama kelas dan value di isi dengan urutan kelas dimulai dari 0
    class_list = {'Nama Kelas 1': 0, 'Nama Kelas 2': 1}
    idx_pred = [0, 0]  # [i.index(max(i)) for i in probs]
    labels = list(class_list.keys())
    return render_template('/201710370311030/result_compare.html', labels=labels,
                           probs=[0, 0], mdl=mdl, run_time=run_time, pred=idx_pred, img=img, result_img=dir_result_img, fid_local=fid_local, fid_global=fid_global, patch_input_manual=is_patch_input)

# Select Model f_nim Brain Tumor Disease


@app.route('/1/select')
def f_201710370311030_select():
    return render_template('/201710370311030/select.html', )


@app.route('/1/pred_select', methods=['POST'])
def f_201710370311030_predict_select():

    chosen_model = request.form['select_model']
    model_dict = {'Nama Model 1':   'static/model/201710370311030/model1.h5',  # Isi dengan Nama model dan path lokasi model disimpan (Pastikan menyimpan model dalam folder /static/model/nim/namamodel.h5)
                  'Nama Model 2':   'static/model/201710370311030/model2.h5',
                  # Jika pakai Json dan weight model saat menyimpan model gunakan kode ini
                  'Nama json Model 1':   ['static/model/201710370311030/js/model_js1.json', 'static/model/201710370311030/js/model_weight_js1.h5'],
                  # Beri kode nama "_js" tanpa petik di akhir nama model
                  'Nama json Model 2':   ['static/model/201710370311030/js/model_js2.json', 'static/model/201710370311030/js/model_weight_js2.h5']
                  }
    if chosen_model in model_dict:
        if "_js" in m:  # dari kode nama model _js selanjutnya program akan membaca model format json dengan block kode dalam if
            json_file = open(model_dict[m][0], 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights(model_dict[m][1])
        else:  # bila nama model tidak mengandung kode nama _js maka model akan di muat menggunakan load_model()
            model = load_model(model_dict[m])
    else:
        model = load_model(model_dict[0])  # load default model

    filename = request.form.get('input_image')

    # preprocessing gambar lakukan sesuai dengan preprocessing yang sama saat proses training
    img = cv2.cvtColor(np.array(Image.open(filename)), cv2.COLOR_BGR2RGB)
    img = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3]
                         else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)

    # mulai prediki
    start = time.time()
    pred = model.predict(img)[0]
    runtimes = round(time.time()-start, 4)  # hitung lama prediksi

    respon_model = [round(elem * 100, 2)
                    for elem in pred]  # hitung nilai prediksi

    return f_201710370311030_predict_result_select(chosen_model, runtimes, respon_model, filename[7:])


@app.route('/1/pred_selects', methods=['POST'])
def f_201710370311030_predicts_select():

    chosen_model = request.form['select_model']
    model_dict = {'Nama Model 1':   'static/model/201710370311030/model1.h5',  # Isi dengan Nama model dan path lokasi model disimpan (Pastikan menyimpan model dalam folder /static/model/nim/namamodel.h5)
                  'Nama Model 2':   'static/model/201710370311030/model2.h5',
                  # Jika pakai Json dan weight model saat menyimpan model gunakan kode ini
                  'Nama json Model 1':   ['static/model/201710370311030/js/model_js1.json', 'static/model/201710370311030/js/model_weight_js1.h5'],
                  # Beri kode nama "_js" tanpa petik di akhir nama model
                  'Nama json Model 2':   ['static/model/201710370311030/js/model_js2.json', 'static/model/201710370311030/js/model_weight_js2.h5']
                  }

    if chosen_model in model_dict:
        if "_js" in chosen_model:  # dari kode nama model _js selanjutnya program akan membaca model format json dengan block kode dalam if
            json_file = open(model_dict[chosen_model][0], 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights(model_dict[chosen_model][1])
        else:  # bila nama model tidak mengandung kode nama _js maka model akan di muat menggunakan load_model()
            model = load_model(model_dict[chosen_model])
    else:
        model = load_model(model_dict[0])  # load default model

    file = request.files["file"]
    file.save(os.path.join('static', 'temp.jpg'))

    # preprocessing gambar lakukan sesuai dengan preprocessing yang sama saat proses training
    img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_BGR2RGB)
    img = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3]
                         else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)

    # mulai prediki
    start = time.time()
    pred = model.predict(img)[0]
    runtimes = round(time.time()-start, 4)  # hitung lama prediksi

    respon_model = [round(elem * 100, 2)
                    for elem in pred]  # hitung nilai prediksi

    return f_201710370311030_predict_result_select(chosen_model, runtimes, respon_model, 'temp.jpg')


def f_201710370311030_predict_result_select(model, run_time, probs, img):
    # isi dengan nama kelas 1 sampai ke n sesuai dengan urutan kelas data pada classification report key di isi dengan nama kelas dan value di isi dengan urutan kelas dimulai dari 0
    class_list = {'Nama Kelas 1': 0, 'Nama Kelas 2': 1}
    idx_pred = probs.index(max(probs))
    labels = list(class_list.keys())
    return render_template('/201710370311030/result_select.html', labels=labels,
                           probs=probs, model=model, pred=idx_pred,
                           run_time=run_time, img=img)


""" Edit End """

if __name__ == "__main__":
    app.run(debug=True)
    # app.run(debug=True, host='0.0.0.0', port=2000)
    # app.run(debug=False, host='0.0.0.0', port=2000, ssl_context=(
    #     '/home/admin/conf/web/ssl.riset.informatika.umm.ac.id.crt', '/home/admin/conf/web/ssl.riset.informatika.umm.ac.id.key'))
