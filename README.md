# 2022년 JBNU(전북대학교) BigData Contest(빅데이터 경진대회)
- 참가자 : 이용환
- 문제 : 1. Text 2. Image 3. IoT(Table Data)
- 결과 : 1. 1등 2. 1등 3. 7등 -> 최종 합산 1등
- 링크 : https://it.jbnu.ac.kr/it/9841/subview.do?enc=Zm5jdDF8QEB8JTJGYmJzJTJGaXQlMkYxNDYyJTJGMjg5MjUzJTJGYXJ0Y2xWaWV3LmRvJTNGcGFnZSUzRDElMjZzcmNoQ29sdW1uJTNEc2olMjZzcmNoV3JkJTNEJUVCJUI5JTg1JUVCJThEJUIwJUVDJTlEJUI0JUVEJTg0JUIwJTI2YmJzQ2xTZXElM0QlMjZiYnNPcGVuV3JkU2VxJTNEJTI2cmdzQmduZGVTdHIlM0QlMjZyZ3NFbmRkZVN0ciUzRCUyNmlzVmlld01pbmUlM0RmYWxzZSUyNnBhc3N3b3JkJTNEJTI2

# 1번 : (Text) 뉴스 주제 분류
- 뉴스 기사 텍스트와 대분류, 소분류가 주어질 때, 각 뉴스 기사를 소분류에 맞게 분류하는 문제
- DistillBert를 Training하여 소분류 데이터-뉴스 기사 본문 데이터를 Classification
- f1 score : 0.78270(Private), 20명 중 1등
- Kaggle Link : https://www.kaggle.com/competitions/jbnu-bigdata2022-text

# 2번 : (Image) 미술품의 양식 분류
- 각 미술품의 이미지를 통해 미술양식을 분류하는 문제
- Image Data Augmentation을 진행 후 Training 진행
- BEiT를 Training하여 미술품 데이터를 Classification
- f1 score : 0.70341(Private), 18명 중 1등
- Kaggle Link : https://www.kaggle.com/competitions/jbnu-bigdata2022-image

# 3번 : (IoT) 에너지 사용량 예측
- 에너지 사용량 데이터(시계열)가 주어질 때(특정 시점의 여러 feature(온도, 습도 ..) 데이터), 특정 시점의 집 안 에너지 사용량을 예측하는 문제
- LSTM을 Training하여 에너지 사용량 예측 -> Baseline 코드에서 Layer 수, Dropout 및 정규화 적용 등 일부만을 수정
- MAE : 0.11445(Private), 20명 중 7등
- Kaggle Link : https://www.kaggle.com/competitions/jbnu-bigdata2022-iot/overview/evaluation
