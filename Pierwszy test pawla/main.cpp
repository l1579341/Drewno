#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace std;

float e = 2.71828182846;
float n = 0.1;
float input[3]; //2 + bias
float weightI_1[3][3];
float layer1[4];
float weight1_O[4];
float output;
float target;


float deltaO;	
float delta1[2];

float sigmoid(float toPow)
{
	return 1/(1+pow(e,-toPow));
}
void generate()
{
	int inputx[2];
	for (int i = 0; i < 2; i++)
	{
		inputx[i] = 2 * (static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
	}
	if ((inputx[0] == 1 && inputx[1] == 1)||(inputx[0] == 0 && inputx[1] == 0)) target = 0;
	else target = float(1);
	input[0] = float(inputx[0]);
	input[1] = float(inputx[1]);
}
void generateBias()
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			weightI_1[j][i] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX))*2-1;
		}
		weight1_O[i] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX))*2-1;
	}
	weight1_O[3] = (static_cast <float> (rand()) / static_cast <float> (RAND_MAX))*2-1;
	input[2] = 1;
	layer1[3] = 1;
	
}
void forward()
{
	for (int i = 0; i < 3; i++)
	{
		float q = 0;
		for (int j = 0; j < 3; j++)
		{
			q += input[j]*weightI_1[j][i];
		}

		layer1[i] = sigmoid(q);
	}
	float p = 0;
	for (int i = 0; i < 4; i++)
	{
		p += layer1[i]*weight1_O[i];
	}
	output = sigmoid(p);
	//output = p;
	//if (p < 0) output = 0;
	//else if (p <= 1) output = p;
	//else output = 1;
}
void back()
{
	deltaO = output*(1 - output)*(output - target);

	for (int i = 0; i<3; i++) //delta
	{	
		float q = 0;
		for (int j = 0; j < 3; j++)
		{
			q += weight1_O[j] * deltaO;
		}
		delta1[i] = layer1[i]*(1 - layer1[i]) * q;
	}
	
	
	for (int i = 0; i < 4; i++) //weight
	{
		weight1_O[i] -= n * layer1[i] * deltaO;
	}	
	for (int i = 0; i < 3; i++) //weight
	{
		for (int j = 0; j < 3; j++)
		{
			weightI_1[j][i] -= n * input[j] * delta1[i];
		}
	}
	
}


int main() 
{	
	srand (static_cast <unsigned> (time(0)));
	generateBias();
	
	int num;
	for (int i = 0; i < 100; i++)
	{
		for (int i = 0; i < 199; i++)
		{
			generate();
			forward();
			back();
			num++;
		}
		generate();
		//cout << "generated: " <<num << endl;
		//cout << "input1: " << input[0] << endl;
		//cout << "input2: " << input[1] << endl;
		//cout << "target: " << target << endl;
		forward();
		//cout << "passed forward" << endl;
		//cout << "output: " << output << endl;
		cout << "loss: " << (target - output) << endl;
		//cout << (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) << endl;
		back();
		//cout << "back propagated" << endl << endl;
		cout << "x: " <<  weight1_O[0] << endl;
		num++;
	}
	
	
	
	return 0;
}

/*
for (int i = 0; i < 3; i++)
	{
		deltaO[i] = output*(1 - output)*(output - target)*layer1[i];
	}
	for (int i = 0; i < 3; i++)
	{
		weight1_O[i] += (-n) * deltaO;
	}	
*/

