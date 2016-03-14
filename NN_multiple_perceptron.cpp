// NN_test.cpp : �R���\�[�� �A�v���P�[�V�����̃G���g�� �|�C���g���`���܂��B
//

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//NN���j�b�g�̐�
#define UNIT_input 2
#define UNIT_hidden 2
#define UNIT_output 1
//���ԑw�̐�
#define NUM_hidden 1

//�w�K�W��
#define epsilon 10//0.5
//�����W���H
#define alpha 0.5
//�P���������֐��i�������֐��j
#define sigmoid(u) (1.0/(1.0 + exp(-1.0 * u)))
//�w�K��
#define T 501
//����
#define RAND ( (double)rand() / (double)RAND_MAX)
inline void InitRand(){ srand((unsigned int)time(NULL)); }

void Initialize(double(*w1)[UNIT_hidden], double(*w2)){

	//���������ݎ����ŃZ�b�g
	InitRand();

	//���͑w->���ԑw

	for (int x = 0; x < (UNIT_input + 1); x++){
		for (int y = 0; y < UNIT_hidden; y++){
			w1[x][y] = RAND;
			printf("WEIGHT_input2hidden[%d][%d] : %.2f\n", x, y, w1[x][y]);
		}
	}

	//���ԑw->�o�͑w
	for (int x = 0; x < (UNIT_hidden + 1); x++){
		//for (int y = 0; y < UNIT_output; y++){
		w2[x] = RAND;
		printf("WEIGHT_hidden2output[%d] : %.2f\n", x, w2[x]);
		//}
	}
}

void Initialize_DELTA_WEIGHT(double(*w1)[UNIT_hidden], double(*w2)){
	//���͑w->���ԑw
	for (int x = 0; x < (UNIT_input + 1); x++){
		for (int y = 0; y < UNIT_hidden; y++){
			w1[x][y] = 0.0;
		}
	}

	//���ԑw->�o�͑w
	for (int x = 0; x < (UNIT_hidden + 1); x++){

		w2[x] = 0.0;

	}
}


int _tmain(int argc, _TCHAR* argv[])
{
	//�P���pXOR�f�[�^
	int input[][UNIT_input + 1] = { { 0, 0, 1 }, { 0, 1, 1 }, { 1, 0, 1 }, { 1, 1, 1 } };	//���͐M��(+1�̓o�C�A�X�p�̉�����)
	int output[] = { 0, 1, 1, 0 };									//���t�M��

	//�����׏d(+1�Ƃ��āA�z��̍Ō�Ƀo�C�A�X(臒l)�̘g��݂���)
	double WEIGHT_input2hidden[UNIT_input + 1][UNIT_hidden];
	double WEIGHT_hidden2output[UNIT_hidden + 1]/*[UNIT_output]*/;
	Initialize(WEIGHT_input2hidden, WEIGHT_hidden2output);

	//�����׏d�C����(��w)
	double DELTA_WEIGHT_input2hidden[UNIT_input + 1][UNIT_hidden];
	double DELTA_WEIGHT_hidden2output[UNIT_hidden + 1]/*[UNIT_output]*/;

	//�C���O�����׏d(OLD_WEIGHT)
	double OLD_WEIGHT_hidden[UNIT_hidden + 1]/*[UNIT_output]*/ = { 0 };

	//�v�Z�p
	double buff;

	//�R���\�[���o�͂̊Ԋu
	const int STEP_SHOW = 10;

	//�w�K�t�F�[�Y
	for (int t = 0; t < T; t++){
		if (t % STEP_SHOW == 0)printf("�y�w�K%d���[�v�ځz\n", t);

		for (int k = 0; k < (sizeof(input) / sizeof(input[0])); k++){

			if (t%STEP_SHOW == 0)printf("����AB : [%d , %d]\t���t�M��y^ : %d\t", input[k][0], input[k][1], output[k]);

			//-----���͑w->���ԑw-----

			//�������u���`�i�������j
			double u_hidden[UNIT_hidden/* + 1*/];
			double y_hidden[UNIT_hidden + 1];
			//for (int count = 0; count < UNIT_hidden ; count++) u_hidden[count] = 0;
			//u_hidden[UNIT_hidden] = 1.0;

			//���͂ƌ����׏d�̐ς̑��a���v�Z���A�������֐���ʂ�
			for (int j = 0; j < UNIT_hidden; j++){
				u_hidden[j] = 0;
				for (int i = 0; i < UNIT_input + 1; i++){
					u_hidden[j] += input[k][i] * WEIGHT_input2hidden[i][j];
				}
				y_hidden[j] = sigmoid(u_hidden[j]);// +WEIGHT_input[UNIT_input][j]);
			}
			y_hidden[UNIT_hidden] = 1;

			if (t%STEP_SHOW == 0)printf("���ԑwU(AB) : [%f , %f]\t", u_hidden[0], u_hidden[1]);

			//for (int count = 0; count < UNIT_hidden; count++) printf("u[%d] : %f\n", count,u[count]);
			//getchar();

			//-----���ԑw->�o�͑w-----

			//�o�͂��`(������)
			double y_output; //�l�b�g���[�N�̏o��


			//���ԑw�̏o�͂ƌ����׏d�̐ς̑��a���v�Z���A�������֐���ʂ�
			/*for (int j = 0; j < UNIT_output; j++){*/
			double u_output = 0;
			for (int j = 0; j < UNIT_hidden + 1; j++){
				u_output += y_hidden[j] * WEIGHT_hidden2output[j];
			}
			y_output = sigmoid(u_output);

			//�덷�֐�
			double E = 0.5 * pow((y_output - (double)output[k]), 2.0);

			if (t % STEP_SHOW == 0)printf("�o��y : %f\t�덷�֐�E : %f\n", y_output, E);
			//}

			//�덷�t�`�d�ɂ��l�b�g���[�N�𒲐�

			//�����׏d�C����(��w)�̏�����
			//Initialize_DELTA_WEIGHT(DELTA_WEIGHT_input2hidden, DELTA_WEIGHT_hidden2output);

			//-----�o�͑w->���ԑw-----

			double TEACHER_SIGNAL = (y_output - output[k]);
			double SIGMOID = y_output * (1.0 - y_output);

#if 1	
			for (int j = 0; j < (UNIT_hidden + 1); j++){
				//���ԑw�̕␳�l
				DELTA_WEIGHT_hidden2output[j] = TEACHER_SIGNAL * SIGMOID * y_hidden[j] /*+ alpha * DELTA_WEIGHT_hidden[j][0]�@<- �w�K�����p�����Ƃ肠���������������̂ŏȗ��@*/;
				//�d�ݍX�V
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
			//-----���ԑw->���͑w-----
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

			//�d�݂̍X�V <-�����Ɏ����ė����OLD_WEIGHT�݂͐��Ȃ��Ă��悭�Ȃ�


		}
	}
	for (int yy = 0; yy <= 10; yy++) {
		for (int xx = 0; xx <= 10; xx++){


			//�������u���`�i�������j
			double u_hidden[UNIT_hidden/* + 1*/];
			double y_hidden[UNIT_hidden + 1];
			//for (int count = 0; count < UNIT_hidden ; count++) u_hidden[count] = 0;
			//u_hidden[UNIT_hidden] = 1.0;

			//���͂ƌ����׏d�̐ς̑��a���v�Z���A�������֐���ʂ�
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

			//if (t%STEP_SHOW == 0)printf("���ԑwU(AB) : [%f , %f]\t", u_hidden[0], u_hidden[1]);

			//for (int count = 0; count < UNIT_hidden; count++) printf("u[%d] : %f\n", count,u[count]);
			//getchar();

			//-----���ԑw->�o�͑w-----

			//�o�͂��`(������)
			double y_output; //�l�b�g���[�N�̏o��


			//���ԑw�̏o�͂ƌ����׏d�̐ς̑��a���v�Z���A�������֐���ʂ�
			/*for (int j = 0; j < UNIT_output; j++){*/
			double u_output = 0;
			for (int j = 0; j < UNIT_hidden + 1; j++){
				u_output += y_hidden[j] * WEIGHT_hidden2output[j];
			}
			y_output = sigmoid(u_output);

			//�덷�֐�
			//double E = 0.5 * pow((y_output - (double)output[k]), 2.0);
			printf("%3.1f ", y_output);
			//if (t % STEP_SHOW == 0)printf("�o��y : %f\t�덷�֐�E : %f\n", y_output, E);
			//}
		}
		puts("");
	}
	return 0;
}

