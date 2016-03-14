// NN_test.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//NNユニットの数
#define UNIT_input 2
#define UNIT_hidden 2
#define UNIT_output 1
//中間層の数
#define NUM_hidden 1

//学習係数
#define epsilon 10//0.5
//加速係数？
#define alpha 0.5
//単調化増加関数（活性化関数）
#define sigmoid(u) (1.0/(1.0 + exp(-1.0 * u)))
//学習回数
#define T 501
//乱数
#define RAND ( (double)rand() / (double)RAND_MAX)
inline void InitRand(){ srand((unsigned int)time(NULL)); }

void Initialize(double(*w1)[UNIT_hidden], double(*w2)){

	//乱数を現在時刻でセット
	InitRand();

	//入力層->中間層

	for (int x = 0; x < (UNIT_input + 1); x++){
		for (int y = 0; y < UNIT_hidden; y++){
			w1[x][y] = RAND;
			printf("WEIGHT_input2hidden[%d][%d] : %.2f\n", x, y, w1[x][y]);
		}
	}

	//中間層->出力層
	for (int x = 0; x < (UNIT_hidden + 1); x++){
		//for (int y = 0; y < UNIT_output; y++){
		w2[x] = RAND;
		printf("WEIGHT_hidden2output[%d] : %.2f\n", x, w2[x]);
		//}
	}
}

void Initialize_DELTA_WEIGHT(double(*w1)[UNIT_hidden], double(*w2)){
	//入力層->中間層
	for (int x = 0; x < (UNIT_input + 1); x++){
		for (int y = 0; y < UNIT_hidden; y++){
			w1[x][y] = 0.0;
		}
	}

	//中間層->出力層
	for (int x = 0; x < (UNIT_hidden + 1); x++){

		w2[x] = 0.0;

	}
}


int _tmain(int argc, _TCHAR* argv[])
{
	//訓練用XORデータ
	int input[][UNIT_input + 1] = { { 0, 0, 1 }, { 0, 1, 1 }, { 1, 0, 1 }, { 1, 1, 1 } };	//入力信号(+1はバイアス用の仮入力)
	int output[] = { 0, 1, 1, 0 };									//教師信号

	//結合荷重(+1として、配列の最後にバイアス(閾値)の枠を設ける)
	double WEIGHT_input2hidden[UNIT_input + 1][UNIT_hidden];
	double WEIGHT_hidden2output[UNIT_hidden + 1]/*[UNIT_output]*/;
	Initialize(WEIGHT_input2hidden, WEIGHT_hidden2output);

	//結合荷重修正量(Δw)
	double DELTA_WEIGHT_input2hidden[UNIT_input + 1][UNIT_hidden];
	double DELTA_WEIGHT_hidden2output[UNIT_hidden + 1]/*[UNIT_output]*/;

	//修正前結合荷重(OLD_WEIGHT)
	double OLD_WEIGHT_hidden[UNIT_hidden + 1]/*[UNIT_output]*/ = { 0 };

	//計算用
	double buff;

	//コンソール出力の間隔
	const int STEP_SHOW = 10;

	//学習フェーズ
	for (int t = 0; t < T; t++){
		if (t % STEP_SHOW == 0)printf("【学習%dループ目】\n", t);

		for (int k = 0; k < (sizeof(input) / sizeof(input[0])); k++){

			if (t%STEP_SHOW == 0)printf("入力AB : [%d , %d]\t教師信号y^ : %d\t", input[k][0], input[k][1], output[k]);

			//-----入力層->中間層-----

			//内部状態uを定義（初期化）
			double u_hidden[UNIT_hidden/* + 1*/];
			double y_hidden[UNIT_hidden + 1];
			//for (int count = 0; count < UNIT_hidden ; count++) u_hidden[count] = 0;
			//u_hidden[UNIT_hidden] = 1.0;

			//入力と結合荷重の積の総和を計算し、活性化関数を通す
			for (int j = 0; j < UNIT_hidden; j++){
				u_hidden[j] = 0;
				for (int i = 0; i < UNIT_input + 1; i++){
					u_hidden[j] += input[k][i] * WEIGHT_input2hidden[i][j];
				}
				y_hidden[j] = sigmoid(u_hidden[j]);// +WEIGHT_input[UNIT_input][j]);
			}
			y_hidden[UNIT_hidden] = 1;

			if (t%STEP_SHOW == 0)printf("中間層U(AB) : [%f , %f]\t", u_hidden[0], u_hidden[1]);

			//for (int count = 0; count < UNIT_hidden; count++) printf("u[%d] : %f\n", count,u[count]);
			//getchar();

			//-----中間層->出力層-----

			//出力を定義(初期化)
			double y_output; //ネットワークの出力


			//中間層の出力と結合荷重の積の総和を計算し、活性化関数を通す
			/*for (int j = 0; j < UNIT_output; j++){*/
			double u_output = 0;
			for (int j = 0; j < UNIT_hidden + 1; j++){
				u_output += y_hidden[j] * WEIGHT_hidden2output[j];
			}
			y_output = sigmoid(u_output);

			//誤差関数
			double E = 0.5 * pow((y_output - (double)output[k]), 2.0);

			if (t % STEP_SHOW == 0)printf("出力y : %f\t誤差関数E : %f\n", y_output, E);
			//}

			//誤差逆伝播によりネットワークを調整

			//結合荷重修正量(Δw)の初期化
			//Initialize_DELTA_WEIGHT(DELTA_WEIGHT_input2hidden, DELTA_WEIGHT_hidden2output);

			//-----出力層->中間層-----

			double TEACHER_SIGNAL = (y_output - output[k]);
			double SIGMOID = y_output * (1.0 - y_output);

#if 1	
			for (int j = 0; j < (UNIT_hidden + 1); j++){
				//中間層の補正値
				DELTA_WEIGHT_hidden2output[j] = TEACHER_SIGNAL * SIGMOID * y_hidden[j] /*+ alpha * DELTA_WEIGHT_hidden[j][0]　<- 学習加速用だがとりあえず実装したいので省略　*/;
				//重み更新
				OLD_WEIGHT_hidden[j] = WEIGHT_hidden2output[j];
				//if (t%STEP_SHOW == 0)printf("DELTA_WEIGHT_hidden[%d][%d] : %f\n", j, i, DELTA_WEIGHT_hidden[j][i]);
				WEIGHT_hidden2output[j] += -epsilon * DELTA_WEIGHT_hidden2output[j];
			}
#endif		
#if 0
			WEIGHT_input2hidden[0][0] = -6.4;
			WEIGHT_input2hidden[1][0] = -6.4;
			WEIGHT_input2hidden[2][0] = -2.6;
			WEIGHT_input2hidden[0][1] = -4.7;
			WEIGHT_input2hidden[1][1] = -4.7;
			WEIGHT_input2hidden[2][1] = -7.0;

			WEIGHT_hidden2output[0] = -9.8;
			WEIGHT_hidden2output[1] =  9.7;
			WEIGHT_hidden2output[2] = 4.6;
#endif
#if 1
			//-----中間層->入力層-----
			for (int j = 0; j < UNIT_hidden; j++){
				//	double TEACHER_SIGNAL = (output[k] - y_output);
				//	double SIGMOID = y_output * (1.0 - y_output);
				for (int i = 0; i < UNIT_input + 1; i++){
					DELTA_WEIGHT_input2hidden[i][j] = TEACHER_SIGNAL * SIGMOID * OLD_WEIGHT_hidden[j] * y_hidden[j] * (1.0 - y_hidden[j]) * input[k][i] /*+ alpha * DELTA_WEIGHT_input[i][j]*/;
					//if (t%STEP_SHOW == 0)printf("DELTA_WEIGHT_input[%d][%d] : %f\n", i, j, DELTA_WEIGHT_input[j][i]);
					WEIGHT_input2hidden[i][j] += -epsilon * DELTA_WEIGHT_input2hidden[i][j];
				}
			}
#endif
			//if (t%STEP_SHOW == 0)printf("\n");
			//getchar();

			//重みの更新 <-ここに持って来ればOLD_WEIGHTは設けなくてもよくなる


		}
	}
	for (int yy = 0; yy <= 10; yy++) {
		for (int xx = 0; xx <= 10; xx++){


			//内部状態uを定義（初期化）
			double u_hidden[UNIT_hidden/* + 1*/];
			double y_hidden[UNIT_hidden + 1];
			//for (int count = 0; count < UNIT_hidden ; count++) u_hidden[count] = 0;
			//u_hidden[UNIT_hidden] = 1.0;

			//入力と結合荷重の積の総和を計算し、活性化関数を通す
			for (int j = 0; j < UNIT_hidden; j++){
				u_hidden[j] = 0;
				//for (int i = 0; i < UNIT_input + 1; i++){
				//	u_hidden[j] += input[k][i] * WEIGHT_input2hidden[i][j];
				//}
				u_hidden[j] += (xx / 10.) * WEIGHT_input2hidden[0][j];
				u_hidden[j] += (yy / 10.) * WEIGHT_input2hidden[1][j];
				u_hidden[j] += (10 / 10.) * WEIGHT_input2hidden[2][j];

				y_hidden[j] = sigmoid(u_hidden[j]);// +WEIGHT_input[UNIT_input][j]);
			}
			y_hidden[UNIT_hidden] = 1;

			//if (t%STEP_SHOW == 0)printf("中間層U(AB) : [%f , %f]\t", u_hidden[0], u_hidden[1]);

			//for (int count = 0; count < UNIT_hidden; count++) printf("u[%d] : %f\n", count,u[count]);
			//getchar();

			//-----中間層->出力層-----

			//出力を定義(初期化)
			double y_output; //ネットワークの出力


			//中間層の出力と結合荷重の積の総和を計算し、活性化関数を通す
			/*for (int j = 0; j < UNIT_output; j++){*/
			double u_output = 0;
			for (int j = 0; j < UNIT_hidden + 1; j++){
				u_output += y_hidden[j] * WEIGHT_hidden2output[j];
			}
			y_output = sigmoid(u_output);

			//誤差関数
			//double E = 0.5 * pow((y_output - (double)output[k]), 2.0);
			printf("%3.1f ", y_output);
			//if (t % STEP_SHOW == 0)printf("出力y : %f\t誤差関数E : %f\n", y_output, E);
			//}
		}
		puts("");
	}
	return 0;
}

