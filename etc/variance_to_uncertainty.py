import numpy as np
from scipy.stats import norm

def calculate_probability_within_range(prediction, variance, lower_bound, upper_bound):
    """
    특정 범위 내에 예측값이 있을 확률을 계산
    
    :param prediction: 모델의 예측값 (평균)
    :param variance: 모델 예측값의 분산
    :param lower_bound: 범위의 하한
    :param upper_bound: 범위의 상한
    :return: 범위 내에 예측값이 있을 확률
    """
    std_dev = np.sqrt(variance)  # 표준편차 계산
    lower_cdf = norm.cdf(lower_bound, loc=prediction, scale=std_dev)
    upper_cdf = norm.cdf(upper_bound, loc=prediction, scale=std_dev)
    return upper_cdf - lower_cdf

def main():
    # 예제 예측값 및 분산
    prediction = 10
    variance = 4  # 예를 들어, 표준편차가 2인 경우

    # 예측값이 ±1 표준편차 범위 내에 있을 확률
    probability_within_1_std_dev = calculate_probability_within_range(
        prediction, variance, prediction - np.sqrt(variance), prediction + np.sqrt(variance)
    )
    print(f"예측값이 ±1 표준편차 범위 내에 있을 확률: {probability_within_1_std_dev:.4f}")

    # 예측값이 ±1.96 표준편차 범위 내에 있을 확률 (95% 신뢰구간)
    probability_within_95_confidence = calculate_probability_within_range(
        prediction, variance, prediction - 1.96 * np.sqrt(variance), prediction + 1.96 * np.sqrt(variance)
    )
    print(f"예측값이 ±1.96 표준편차 범위 내에 있을 확률 (95% 신뢰구간): {probability_within_95_confidence:.4f}")

if __name__ == "__main__":
    main()
