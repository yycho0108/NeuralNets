#define ARMA_DONT_USE_CXX11
#include <armadillo>
#include <vector>
#include <iostream>

#include <functional>
#include <ctime>
#include <random>


using namespace arma;
using namespace std;

vec sigmoid_approx(vec& v){
	return 0.5 * (v / (1 + abs(v)) + 1);
}

vec sigmoidPrime_approx(vec& v){
	vec tmp = abs(v) + 1;
	return 0.5 / (tmp%tmp);
}

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

std::string f2h(const float& f){
	char str[9] = {};
	snprintf(str, 9, "%x", *(unsigned int*)(&f));
	return str;
}

void print_hex(const arma::mat& m){
	for(unsigned int i=0; i<m.n_rows; ++i){
		for(unsigned int j=0; j<m.n_cols;++j){
			std::cout << f2h(m(i,j)) << std::endl;
		}
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
	//_O = sigmoid(_I);
	_O = sigmoid_approx(_I);
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
	void print();
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
		std::cout << "----------------" << std::endl;
		std::cout << "[W]" << std::endl;
		print_hex(W[i-1]);
		std::cout << "[I]" << std::endl;
		print_hex(L[i-1].O());
		std::cout << "[O]" << std::endl;
		print_hex(W[i-1]*L[i-1].O());
		std::cout << "[B]" << std::endl;
		print_hex(B[i-1]);
		//
		L[i].transfer(W[i-1]*L[i-1].O() + B[i-1]);	
		//
		std::cout << "[O_2]" << std::endl;
		print_hex(W[i-1]*L[i-1].O() + B[i-1]);
		std::cout << "[Y]" << std::endl;
		print_hex(L[i].O());
	}
	return arma::conv_to<std::vector<double>>::from(L.back().O());
}
void Net::BP(std::vector<double> Y){
	L.back().G() = vec(Y) - L.back().O();
	for(size_t i = t.size()-2;i>=1;--i){
		//L[i].G() = W[i].t() * L[i+1].G() % sigmoidPrime(L[i].O(),true);
		L[i].G() = W[i].t() * L[i+1].G() % sigmoidPrime_approx(L[i].I());
	}
	for(size_t i=1;i<t.size();++i){
		W[i-1] += 0.6 * L[i].G() * L[i-1].O().t();
		B[i-1] += 0.6 * L[i].G();
	}
}

void Net::print(){
	std::cout << W[0] << std::endl;
	for(auto& w : W){
		std::cout << "--------" << std::endl;
		print_hex(w);
	}

	for(auto& b : B){
		std::cout << "--------" << std::endl;
		print_hex(b);
	}
}


double randNum(){
	static auto _randNum = std::bind(std::uniform_real_distribution<double>(0.0,1.0),std::default_random_engine(2)); //random
	return _randNum();
}

void XOR_GEN(std::vector<double>& X, std::vector<double>& Y){
	X[0] = randNum()>0.5?1:0;
	X[1] = randNum()>0.5?1:0;
	Y[0] = int(X[0]) ^ int(X[1]);
}

int main(int argc, char* argv[]){
	arma_rng::set_seed(0);
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
		//print_hex(net.FF(X));
	}

	net.print();
}
