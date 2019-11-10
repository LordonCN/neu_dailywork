/*动态规划类型
爬台阶：递归
矩阵最短路径：遍历
*/

#include <iostream>

using namespace std;

int fac(int n) {//递归
    if(n == 1) 
        return 1;
    else
        return (long int)(n * fac(n-1));
}

int fac_stop(int n,int stop) {//递归
    if(n == stop) 
        return 1;
        
    else
        // cout<<n<<endl;
        return (long int)(n * fac_stop(n-1,stop));
}

int main()
{
    while(1)
    {
        unsigned long int ThereAreWaysToclimb=0;
        int n=0 ; //HowManyStairs
        int i;
        int step_1=1;
        int step_2=2;
        int max_step_2_is=0;
        cout<< "pls input stairs"<<endl;
        cin  >> n ;//input how many stairs
        /*
        分析： 通过计算2的个数可以得出有几种方案
        */
        if (n==1)
            ThereAreWaysToclimb=1;
        else if (n%2==0)
            {
                max_step_2_is = n/2;
                for (int i=1;i<max_step_2_is;i++) //有i步是2的时候
                {
                    ThereAreWaysToclimb=ThereAreWaysToclimb+fac_stop((n-i),n-2*i)/fac_stop(i,1);
                    // ThereAreWaysToclimb=ThereAreWaysToclimb+(fac(n-i))/(fac(i)*fac(n-2*i));
                }
                // cout<<"偶数楼梯"<<endl;
                ThereAreWaysToclimb= ThereAreWaysToclimb+2 ;//加上两端
            }
        else 
            {
                max_step_2_is = n/2;
                for (int i=1;i<=max_step_2_is;i++)
                {
                    // cout<<"------------"<< i<<"------------"<<endl;
                    // fac_stop((n-i),n-2*i);
                    ThereAreWaysToclimb=ThereAreWaysToclimb+fac_stop((n-i),n-2*i)/fac_stop(i,1);
                }
                // cout<<"奇数楼梯"<<endl;
                ThereAreWaysToclimb= ThereAreWaysToclimb+1 ;
            }
        
        cout  << ThereAreWaysToclimb << endl;
    }
}

