from utils.random5Foldstrain import *

seed_torch(2021)
model = "mobileLarge"
manyTimesTrain(model, config.manyTimesTrainFirstPath, times=5, modelName="mobileBigBig")