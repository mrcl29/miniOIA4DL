from modules.conv2d import Conv2D
from modules.relu import ReLU
from modules.maxpool2d import MaxPool2D
from modules.flatten import Flatten
from modules.dense import Dense
from modules.softmax import Softmax
from models.basemodel import BaseModel
from modules.batchnorm import BatchNorm2D
from modules.dropout import Dropout


class OIANET_CIFAR100(BaseModel):
    def __init__(self, conv_algo=0, pool_algo=0, use_gpu=False):
        print("Building OIANet for CIFAR-100")
        layers = [
            Conv2D(3, 32, kernel_size=3, stride=1, padding=1, conv_algo=conv_algo, use_gpu=use_gpu),
            BatchNorm2D(32, use_gpu=use_gpu),
            ReLU(use_gpu=use_gpu),
            MaxPool2D(kernel_size=2, stride=2, pool_algo=pool_algo, use_gpu=use_gpu),

            Conv2D(32, 64, kernel_size=3, stride=1, padding=1, conv_algo=conv_algo, use_gpu=use_gpu),
            BatchNorm2D(64, use_gpu=use_gpu),
            ReLU(use_gpu=use_gpu),
            MaxPool2D(kernel_size=2, stride=2, pool_algo=pool_algo, use_gpu=use_gpu),

            Conv2D(64, 128, kernel_size=3, stride=1, padding=1, conv_algo=conv_algo, use_gpu=use_gpu),
            BatchNorm2D(128, use_gpu=use_gpu),
            ReLU(use_gpu=use_gpu),
            MaxPool2D(kernel_size=2, stride=2, pool_algo=pool_algo, use_gpu=use_gpu),

            Flatten(use_gpu=use_gpu),
            Dense(128 * 4 * 4, 256, use_gpu=use_gpu),
            ReLU(use_gpu=use_gpu),
            Dropout(0.5, use_gpu=use_gpu),

            Dense(256, 100, use_gpu=use_gpu),
            Softmax(use_gpu=use_gpu),
        ]

        super().__init__(layers)