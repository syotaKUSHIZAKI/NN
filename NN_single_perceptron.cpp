// NN_SINGLE.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//NNユニットの数
#define UNIT_input 2
#define UNIT_output 1

//学習係数
#define epsilon 0.02
//加速係数？
#define alpha 0.2
//単調化増加関数（活性化関数）
#define sigmoid(u) (1.0/(1.0 + exp(-1.0 * u)))
//学習回数
#define T 10000
//乱数
#define RAND ((double)rand() / (double)RAND_MAX)
inline void InitRand(){ srand((unsigned int)time(NULL)); }

void Initialize_WEIGHT(double(*w1)){

	//乱数を現在時刻でセット
	InitRand();

	//入力層->出力層

	for (int x = 0; x < (UNIT_input + 1); x++){
		
		w1[x] = RAND;
		printf("WEIGHT_input[%s] : %f\n", x == 0 ? "A" : "B", w1[x]);
		
	}
}


int _tmain(int argc, _TCHAR* argv[])
{
	//訓練用データ
	int input[][UNIT_input] = { { 0, 0 },
								{ 0, 1 },
								{ 1, 0 },
								{ 1, 1 } };
	int output[] = { 0, 1, 1, 1 };

	//入力信号
	int SIGNAL_input[UNIT_input + 1];

	//教師信号
	int SIGNAL_teacher;

	//結合荷重
	double WEIGHT_input[UNIT_input + 1];
	Initialize_WEIGHT(WEIGHT_input);

	//結合荷重修正量(Δw)
	double DELTA_WEIGHT_input[UNIT_input+1];

	//コンソール出力の間隔
	const int STEP_SHOW = 500;

	for (int t = 0 ; t < T; t++){
		if (t % STEP_SHOW == 0)printf("【学習%dループ目】\n", t);

		for (int k = 0; k < (sizeof(input) / sizeof(input[0])); k++){

			//-----入力層->出力層

			//入力信号をセット
			for (int i = 0; i < UNIT_input; i++) SIGNAL_input[i] = input[k][i];
			SIGNAL_input[UNIT_input] = 1.0;
			if (t%STEP_SHOW == 0)printf("入力AB : [%d , %d]\t教師信号y^ : %d\t", SIGNAL_input[0], SIGNAL_input[1], output[k]);

			//教師信号をセット
			SIGNAL_teacher = output[k];

			//内部状態uを定義
			double u;
			u = 0.0;

			//入力と結合荷重の積の総和を計算
			for (int i = 0; i < (UNIT_input + 1); i++){
				u += SIGNAL_input[i] * WEIGHT_input[i];
			}

			//活性化関数を通し、出力yを求める
			double y;
			y = sigmoid(u);
			if (t%STEP_SHOW == 0) printf("出力y : %f\t", y);

			//誤差関数
			double E = 0.5 * pow((SIGNAL_teacher - y), 2.0);
			if (t%STEP_SHOW == 0) printf("誤差関数E : %f\n", E);

			//出力層->入力層
			double TEACHER_SIGNAL = (SIGNAL_teacher - y);
			double SIGMOID = y * (1.0 - y);

			for (int i = 0; i < (UNIT_input + 1); i++){
				//結合荷重の補正値
				DELTA_WEIGHT_input[i] = TEACHER_SIGNAL * SIGMOID * SIGNAL_input[i];
				//結合荷重更新
				WEIGHT_input[i] += DELTA_WEIGHT_input[i];
			}
		}
	}


	return 0;
}

