#ifndef MTHREAD_H
#define MTHREAD_H
#include <boost/function.hpp>
#include <boost/thread.hpp>

class MThread
{
public:
	MThread(){t = 0;}
	~MThread(){if (t) delete t;}
	void start(boost::function< void()> functor){
		if (!t){
			t = new boost::thread(functor);
		}
	}

	void join(){
		if (t){
			t->join();
			delete t;

			t = 0;
		}
	}
	boost::thread *t;
};

#endif