/*
Return error insert result of data
*/

float insert_float(float data, int bit){
    int *p=(void*)&data;
    (*p)^=1<<bit;
    return data;
}

#include <stdio.h>
int main(){
    float x;
    scanf("%f",&x);
    while (1){
        int bit; scanf("%d",&bit);
        x=insert_float(x, bit);
        printf("%e\n", x);
    }
}
