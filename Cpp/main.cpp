#include <armadillo>
#include <vector>
#include <iostream>

#include <functional>
#include <ctime>
#include <random>


using namespace arma;
using namespace std;

double sigmoid(double x){
	return 1.0/(1.0 + exp(-x));
}
vec sigmoid(vec& v){
	return 1.0/(1.0 + exp(-v));
}
vec sigmoidPrime(vec& v, bool sig){
	if(sig)
		return v % (1.0-v);
	else{
		vec s = sigmoid(v);
		return s % (1.0-s);
	}
}

class Layer{
private:
	int n; //size
	vec _I,_O,_G;
public:
	Layer(int n);
	~Layer();
	void transfer(vec);
	vec& I();
	vec& O();
	vec& G();
	int size();
};

Layer::Layer(int n):n(n){
	_I.set_size(n);
	_O.set_size(n);
	_G.set_size(n);
}
Layer::~Layer(){

}

void Layer::transfer(vec v){
	_I.swap(v);
	//_O = _I;
	_O = sigmoid(_I);
	//cout << "I" << arma::size(_I) << endl;
	//cout << "O" << arma::size(_O) << endl;
//	_O.for_each([](mat::elem_type& val){val = sigmoid(val);});
	//return _O;
	//_I.for_each([]())
}
vec& Layer::I(){
	return _I;
}
vec& Layer::O(){
	return _O;
}
vec& Layer::G(){
	return _G;
}
class Net{
private:
	std::vector<int> t; //topology
	std::vector<mat> W;
	std::vector<Layer> L; //layers
	std::vector<vec> B; //biases
public:
	std::vector<double> FF(std::vector<double> X);
	void BP(std::vector<double> Y);
	Net(std::vector<int> t);
};
Net::Net(std::vector<int> t):t(t){
	for(size_t i=1;i<t.size();++i){
		W.push_back(arma::randn<mat>(t[i],t[i-1]));
		B.push_back(arma::randn<vec>(t[i]));
	}
	for(auto& e : t){
		L.push_back(Layer(e));
	}
}
std::vector<double> Net::FF(std::vector<double> X){
	L.front().O() = X;
	for(size_t i=1;i<t.size();++i){
		L[i].transfer(W[i-1]*L[i-1].O() + B[i-1]);	
	}
	return arma::conv_to<std::vector<double>>::from(L.back().O());
}
void Net::BP(std::vector<double> Y){
	L.back().G() = vec(Y) - L.back().O();
	for(size_t i = t.size()-2;i>=1;--i){
		L[i].G() = W[i].t() * L[i+1].G() % sigmoidPrime(L[i].O(),true);
	}
	for(size_t i=1;i<t.size();++i){
		W[i-1] += 0.6 * L[i].G() * L[i-1].O().t();
		B[i-1] += 0.6 * L[i].G();
	}
}

double randNum(){
	static auto _randNum = std::bind(std::uniform_real_distribution<double>(0.0,1.0),std::default_random_engine(time(0))); //random
	return _randNum();
}

void XOR_GEN(std::vector<double>& X, std::vector<double>& Y){
	X[0] = randNum()>0.5?1:0;
	X[1] = randNum()>0.5?1:0;
	Y[0] = int(X[0]) ^ int(X[1]);
}

int main(int argc, char* argv[]){
	int lim = 1000;
	if(argc != 1){
		lim = std::atoi(argv[1]);
	}
	std::vector<int> t({2,4,1});
	Net net(t);
	std::vector<double> X(2);
	std::vector<double> Y(1);

	auto start = clock();
	for(int i=0;i<lim;++i){
		XOR_GEN(X,Y);
		net.FF(X);
		net.BP(Y);
	}
	auto end = clock();
	printf("Took %f seconds", float(end-start)/CLOCKS_PER_SEC);
	for(int i=0;i<10;++i){
		XOR_GEN(X,Y);
		std::cout << X[0] << ',' << X[1] << ':' <<  Y[0] << '|' << net.FF(X)[0] << std::endl;
	}
}
