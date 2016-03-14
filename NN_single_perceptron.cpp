// NN_SINGLE.cpp : �R���\�[�� �A�v���P�[�V�����̃G���g�� �|�C���g���`���܂��B
//

#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//NN���j�b�g�̐�
#define UNIT_input 2
#define UNIT_output 1

//�w�K�W��
#define epsilon 0.02
//�����W���H
#define alpha 0.2
//�P���������֐��i�������֐��j
#define sigmoid(u) (1.0/(1.0 + exp(-1.0 * u)))
//�w�K��
#define T 10000
//����
#define RAND ((double)rand() / (double)RAND_MAX)
inline void InitRand(){ srand((unsigned int)time(NULL)); }

void Initialize_WEIGHT(double(*w1)){

	//���������ݎ����ŃZ�b�g
	InitRand();

	//���͑w->�o�͑w

	for (int x = 0; x < (UNIT_input + 1); x++){
		
		w1[x] = RAND;
		printf("WEIGHT_input[%s] : %f\n", x == 0 ? "A" : "B", w1[x]);
		
	}
}


int _tmain(int argc, _TCHAR* argv[])
{
	//�P���p�f�[�^
	int input[][UNIT_input] = { { 0, 0 },
								{ 0, 1 },
								{ 1, 0 },
								{ 1, 1 } };
	int output[] = { 0, 1, 1, 1 };

	//���͐M��
	int SIGNAL_input[UNIT_input + 1];

	//���t�M��
	int SIGNAL_teacher;

	//�����׏d
	double WEIGHT_input[UNIT_input + 1];
	Initialize_WEIGHT(WEIGHT_input);

	//�����׏d�C����(��w)
	double DELTA_WEIGHT_input[UNIT_input+1];

	//�R���\�[���o�͂̊Ԋu
	const int STEP_SHOW = 500;

	for (int t = 0 ; t < T; t++){
		if (t % STEP_SHOW == 0)printf("�y�w�K%d���[�v�ځz\n", t);

		for (int k = 0; k < (sizeof(input) / sizeof(input[0])); k++){

			//-----���͑w->�o�͑w

			//���͐M�����Z�b�g
			for (int i = 0; i < UNIT_input; i++) SIGNAL_input[i] = input[k][i];
			SIGNAL_input[UNIT_input] = 1.0;
			if (t%STEP_SHOW == 0)printf("����AB : [%d , %d]\t���t�M��y^ : %d\t", SIGNAL_input[0], SIGNAL_input[1], output[k]);

			//���t�M�����Z�b�g
			SIGNAL_teacher = output[k];

			//�������u���`
			double u;
			u = 0.0;

			//���͂ƌ����׏d�̐ς̑��a���v�Z
			for (int i = 0; i < (UNIT_input + 1); i++){
				u += SIGNAL_input[i] * WEIGHT_input[i];
			}

			//�������֐���ʂ��A�o��y�����߂�
			double y;
			y = sigmoid(u);
			if (t%STEP_SHOW == 0) printf("�o��y : %f\t", y);

			//�덷�֐�
			double E = 0.5 * pow((SIGNAL_teacher - y), 2.0);
			if (t%STEP_SHOW == 0) printf("�덷�֐�E : %f\n", E);

			//�o�͑w->���͑w
			double TEACHER_SIGNAL = (SIGNAL_teacher - y);
			double SIGMOID = y * (1.0 - y);

			for (int i = 0; i < (UNIT_input + 1); i++){
				//�����׏d�̕␳�l
				DELTA_WEIGHT_input[i] = TEACHER_SIGNAL * SIGMOID * SIGNAL_input[i];
				//�����׏d�X�V
				WEIGHT_input[i] += DELTA_WEIGHT_input[i];
			}
		}
	}


	return 0;
}

