from keras.optimizers import SGD,Adam
from unet import unet
from Data import *
from keras.models import load_model

def main():
    model = unet()
    model.load_weights('modelsave/model_weights_iou222test_agian.h5')

    print("training finish")

    x_test,y_test = getdata(1)
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
