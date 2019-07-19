from keras.optimizers import SGD,Adam
from unet import unet
# from unet_model import Unet
# from VGG_Unet import VGG_Unet
from ResUnet1 import ResUnet
from Data import *
from keras.models import load_model

def main():
    model = ResUnet()
    # model.load_weights('modelsave/my_model_weights.h5')

    print("loading data")
    x_train,y_train = get_train_data()
    x_valid,y_valid = get_dev_data()
    x_test,y_test = get_test_data()
    # x_test,name = predicted_data()
    print("start training")
    model.fit(x_train,y_train,validation_data = [x_valid,y_valid],epochs=93,batch_size=32,verbose=1)

    # model = load_model('modelsave/my_model.h5')
    # model.fit_generator(generator_data(4,0),steps_per_epoch=3200,epochs=10)
    print("training finish")

    score = model.evaluate(x_test,y_test,batch_size = 32,verbose = 1)
    # score = model.evaluate_generator(generator_data(4,1),steps=200)
    print("score = ",score)

    # print("predict")
    # data_test,name = predicted_data()
    # result = model.predict(data_test,batch_size = 32)
    # result = model.predict_on_batch(data);
    # resultsave("predicts",result,name)
    # maskcopy(name)

    model.save("modelsave/model_ResUnet_norm_iou_dateaug.h5")
    model.save_weights("modelsave/model_weights_ResUnet_norm_iou_dateaug.h5")


if __name__ == "__main__":
    main()
