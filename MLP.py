'''
Run on Google Colab
MLP
MNIST
'''

from google.colab import drive

drive.mount('/content/mai_drive') 

import time
import math
from torch import nn, optim, cuda
from torch.utils import data
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch


class MLP(nn.Module):

    ''' 
    [네트워크 정의]
    MLP
    nn.Module : 일반적으로 필요한 변수, 메서드들이 있다.
    '''

    def __init__(self):
        super().__init__()

        # 784 : 1 x 28 x 28
        self.l1 = nn.Linear(784, 520) # 첫 번째 레이어의 input feature는 784개이고 output feature는 520개이다.
        self.l2 = nn.Linear(520, 320) # 두 번째 레이어의 input feature는 520개이고 output feature는 320개이다.
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)

    # 그레이 이미지 : 1 x w x h (mnist)
    # 컬러 이미지 : 3 x w x h
    # n : batch size
    # Flatten : 4차원짜리 텐서를 2차원짜리의 텐서로 변환한다.
    def forward(self, inputData):
        # inputData = ( n, 1, 28, 28 )
        # -1의 의미는 "python! 너가 알아서 해줘"라는 의미다.
        # 즉, (-1, 784)의 의미는 앞에는 적당하게 맞추고 뒤에는 784로 맞춘다라는 의미이다.
        # 즉, 여기서는 -1은 알아서 batch size(64)로 변환된다. 아래의 주석에서 n은 batch size를 의미함 (n이라고 한 이유는 batch size가 변할 수도 있음을 강조하기 위해 표기함)
        neurons1 = inputData.view(-1, 784) # Flatten the data ( n, 1, 28, 28 ) -> (n, 784), view : numpy에서의 reshape 메서드
        neurons2 = F.relu(self.l1(neurons1))
        neurons3 = F.relu(self.l2(neurons2))
        neurons4 = F.relu(self.l3(neurons3))
        neurons4 = F.relu(self.l4(neurons4))
        outputData = self.l5(neurons4)
        return outputData


class Machine:

    '''
    dataset : mnist
    model : MLP
    '''

    BASE_DIR = "/content/mai_drive/MyDrive/deep_learning"
    DATASETS_DIR = f'{BASE_DIR}/datasets'


    def __init__(self, batch_size=64, epoch_size=1):
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.device = 'cuda' if cuda.is_available() else 'cpu' 
        self.get_data()
        self.model = MLP()
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss() 
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)


    def get_data(self):

        # MNIST dataset 다운로드 받아서 원하는 폴더에 저장한다.
        self.train_dataset = datasets.MNIST(
            root=f'{Machine.DATASETS_DIR}/mnist_data/',
            train=True,
            transform=transforms.ToTensor(),
            download=True
        )

        self.test_dataset = datasets.MNIST(
            root=f'{Machine.DATASETS_DIR}/mnist_data/',
            train=False,
            transform=transforms.ToTensor()
        )

        # Data loader : 배치 사이즈에 맞게 데이터를 로드한다.
        ## 데이터셋을 불러와 학습에 활용할 수 있도록 원하는 크기의 미니배치로 나누어 읽어 들인다.
        ## 배치 사이즈가 4라는 의미는 이미지 4개를 한번에 넣어서 에러를 한번에 계산하고 그걸 이용해서 모두가 weigth 업데이트를 하는 것
        ## 배치 사이즈가 1이라는 의미는 이미지 매 한 장마다 weight 업데이트를 하는 것이다.
        ## shuffle 옵션 : trian 데이터 셋에서 데이터를 한 번 불러올 때마다 데이터 순서를 랜덤하게 섞어서 불러올 것인지 아닌지

        self.train_loader = data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.test_loader = data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

    # 데이터를 이용하여 모델 학습
    def train(self, epoch):
        # [epoch]
        # 한 번의 epoch는 인공 신경망에서 전체 데이터 셋에 대해 forward pass / backward pass 과정을 거친 것을 말함. 
        # 즉, 전체 데이터 셋에 대해 한 번 학습을 완료한 상태
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad() # 각각의 weight값을 업데이트할 때마다 초기화한다.
            output = self.model(data) # model.forward
            loss = self.criterion(output, target) # 에러 계산
            loss.backward() # weight update
            self.optimizer.step()

            if batch_idx % 10 == 0: # batch 가 10번 진행될 때마다 출력 결과 확인
                print((f'Train Epoch : {epoch} | Batch Status : {batch_idx*len(data)}/{len(self.train_loader.dataset)}'
                    f'({100. * batch_idx / len(self.train_loader)}%) | Loss : {loss.item()}'))
    
    # 학습된 모델의 성능 체크
    # TestData는 TrainData와 독립적이여야 한다.
    # Test는 Forward만 진행
    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            test_loss += self.criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max, 가장 높은 확률을 class하나를 정답으로 선언
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            test_loss /= len(self.test_loader.dataset)
            print(f'\nTest set: Accuracy : {correct}/{len(self.test_loader.dataset)}'
                f'({100. * correct / len(self.test_loader.dataset):.0f}%)')
    

    def learning(self):
        since = time.time()
        for epoch in range(1, self.epoch_size + 1):
            epoch_start = time.time()
            self.train(epoch)
            m, s = divmod(time.time() - epoch_start, 60)
            print(f'Training time: {m:.0f}m {s:.0f}s')
            
        self.test()

        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Tesing time: {m:.0f}m {s:.0f}s')

        m, s = divmod(time.time() - epoch_start, 60)
        print(f'Total time : {m:.0f}m {s: .0f}s \nModel was trained on {self.device}!')


if __name__ == "__main__":
    machine = Machine(batch_size=64, epoch_size=10)
    machine.learning()
