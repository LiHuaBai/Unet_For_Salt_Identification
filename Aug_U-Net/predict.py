from keras.optimizers import SGD,Adam
from ResUnet1 import ResUnet
from unet import unet
from Data import *
from keras.models import load_model

def main():
    model = ResUnet()
    model.load_weights('modelsave/model_weights_ResUnet_norm_iou_dateaug.h5')

    print("training finish")

    x_test,y_test = get_test_data()
    score = model.evaluate(x_test,y_test,batch_size = 32,verbose = 1)
    # score = model.evaluate_generator(generator_data(4,1),steps=200)
    print("score = ",score)

    print("predict")
    data_test,name = predicted_data()
    result = model.predict(data_test,batch_size = 32)
    #
    resultsave("predicts",result,name)



if __name__ == "__main__":
    main()
