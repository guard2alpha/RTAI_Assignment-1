# RTAI Assignment 1: Adversarial Attacks

MNIST와 CIFAR-10 데이터셋을 활용하여 적대적 Adversarial Attacks 알고리즘을 구현 및 성능을 평가하기 위해 구성되었습니다. 코드는 기능별로 모듈화되어 있습니다.

## 📂 Repository Structure

* `datasets.py` : 데이터셋(MNIST, CIFAR-10) 다운로드
* `models.py` : 타겟 모델 아키텍처 (MNIST용 Simple CNN, CIFAR-10용 Pre-trained ResNet20 및 정규화 wrapper)
* `attacks.py` : 4가지 Adversarial Attack 알고리즘 구현 (Targeted/Untargeted FGSM, Targeted/Untargeted PGD)
* `test.py` : 모델 학습, 공격 평가 실행, 결과 시각화 및 성공률을 계산하는 메인 스크립트
* `requirements.txt` : 환경

## 🚀 How to Run