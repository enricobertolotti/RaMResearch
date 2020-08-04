from RaMCode.DeepLearning.Data import *
from RaMCode.DeepLearning.Model import *
from keras.callbacks import ModelCheckpoint


###################### File Path Variables
data_folder = "/Users/enricobertolotti/PycharmProjects/BScAssignment/RaMData/ML_Training_Data"
image_folder = "Images"
mask_folder = "Masks"
output_folder = "Output"

################ ML Training Arguments
data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')

#  os.environ["CUDA_VISIBLE_DEVICES"] = "0"


########## Main Script
myGene = trainGenerator(2, data_folder, image_folder, mask_folder, data_gen_args, save_to_dir=None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])

testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene, 30, verbose=1)
saveResult("data/membrane/test", results)
