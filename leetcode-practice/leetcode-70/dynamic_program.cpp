/*动态规划类型
爬台阶：递归
矩阵最短路径：遍历
*/


#include <iostream>
using namespace std;

# define N  200
int pb[N];

int dynamic_program(int n) {
    if((n == 1)||(n == 2)) 
        return n;
    pb[n-1] = dynamic_program(n-1);
    pb[n-2] = dynamic_program(n-2);
    pb[n] = pb[n-1]+pb[n-2];
    return pb[n];
}

int main()
{
    while(1)
    {
        int n;
        cout<<"pls input n:"<<endl;
        cin>> n;

        pb[N-1] = dynamic_program(n);
        cout<< pb[N-1]<<endl;
    }
}
