ln=3; nT=50; tn=49;
X=zeros(tn,9); X2=zeros(tn,9); X3=zeros(tn,9); X4=zeros(tn,9);
X5=zeros(tn,9); X6=zeros(tn,9); X7=zeros(tn,9); X8=zeros(tn,9);
Y=dJ;
nS=300;
Yo1=OilTraj1'; Yo2=OilTraj2'; Yo3=OilTraj3'; Yo4=OilTraj4';
Yo5=OilTraj5'; Yo6=OilTraj6'; Yo7=OilTraj7'; Yo8=OilTraj8';
Yw1=WProTraj1'; Yw2=WProTraj2'; Yw3=WProTraj3'; Yw4=WProTraj4';
Yw5=WProTraj5'; Yw6=WProTraj6'; Yw7=WProTraj7'; Yw8=WProTraj8';

U1=U1Traj'; U2=U2Traj'; U3=U3Traj'; U4=U4Traj';
U5=U5Traj'; U6=U6Traj'; U7=U7Traj'; U8=U8Traj';
Theta=[]; Theta2=[];


while ln<=nT,
    sn=4; 
        
    while sn<=nS,
        
        X(ln-1,1)= X(ln-1,1)+ Yo1(ln-1, sn)*dU1(ln-1,sn);
        X(ln-1,2)= X(ln-1,2)+ Yw1(ln-1, sn)*dU1(ln-1,sn);
        X(ln-1,3)= X(ln-1,3)+ Yo1(ln-1, sn-1)*dU1(ln-1,sn);
        X(ln-1,4)= X(ln-1,4)+ Yw1(ln-1, sn-1)*dU1(ln-1,sn);
        X(ln-1,5)= X(ln-1,5)+ Yo1(ln-1, sn-2)*dU1(ln-1,sn);
        X(ln-1,6)= X(ln-1,6)+ Yw1(ln-1, sn-2)*dU1(ln-1,sn);
        X(ln-1,7)= X(ln-1,7)+ Yo1(ln-1, sn-3)*dU1(ln-1,sn);
        X(ln-1,8)= X(ln-1,8)+ Yw1(ln-1, sn-3)*dU1(ln-1,sn);
        X(ln-1,9)= X(ln-1,9)+ U1(ln-1, sn)*dU1(ln-1,sn);
        
        
        X2(ln-1,1)= X2(ln-1,1)+ Yo2(ln-1, sn)*dU2(ln-1,sn);
        X2(ln-1,2)= X2(ln-1,2)+ Yw2(ln-1, sn)*dU2(ln-1,sn);
        X2(ln-1,3)= X2(ln-1,3)+ Yo2(ln-1, sn-1)*dU2(ln-1,sn);
        X2(ln-1,4)= X2(ln-1,4)+ Yw2(ln-1, sn-1)*dU2(ln-1,sn);
        X2(ln-1,5)= X2(ln-1,5)+ Yo2(ln-1, sn-2)*dU2(ln-1,sn);
        X2(ln-1,6)= X2(ln-1,6)+ Yw2(ln-1, sn-2)*dU2(ln-1,sn);
       X2(ln-1,7)= X2(ln-1,7)+ Yo2(ln-1, sn-3)*dU2(ln-1,sn);
        X2(ln-1,8)= X2(ln-1,8)+ Yw2(ln-1, sn-3)*dU2(ln-1,sn);
        X2(ln-1,9)= X2(ln-1,9)+ U2(ln-1, sn)*dU2(ln-1,sn);
        
        
        X3(ln-1,1)= X3(ln-1,1)+ Yo3(ln-1, sn)*dU3(ln-1,sn);
        X3(ln-1,2)= X3(ln-1,2)+ Yw3(ln-1, sn)*dU3(ln-1,sn);
        X3(ln-1,3)= X3(ln-1,3)+ Yo3(ln-1, sn-1)*dU3(ln-1,sn);
        X3(ln-1,4)= X3(ln-1,4)+ Yw3(ln-1, sn-1)*dU3(ln-1,sn);
        X3(ln-1,5)= X3(ln-1,5)+ Yo3(ln-1, sn-2)*dU3(ln-1,sn);
        X3(ln-1,6)= X3(ln-1,6)+ Yw3(ln-1, sn-2)*dU3(ln-1,sn);
      X3(ln-1,7)= X3(ln-1,7)+ Yo3(ln-1, sn-3)*dU3(ln-1,sn);
        X3(ln-1,8)= X3(ln-1,8)+ Yw3(ln-1, sn-3)*dU3(ln-1,sn);
        X3(ln-1,9)= X3(ln-1,9)+ U3(ln-1, sn)*dU3(ln-1,sn);

        
        X4(ln-1,1)= X4(ln-1,1)+ Yo4(ln-1, sn)*dU4(ln-1,sn);
        X4(ln-1,2)= X4(ln-1,2)+ Yw4(ln-1, sn)*dU4(ln-1,sn);
        X4(ln-1,3)= X4(ln-1,3)+ Yo4(ln-1, sn-1)*dU4(ln-1,sn);
        X4(ln-1,4)= X4(ln-1,4)+ Yw4(ln-1, sn-1)*dU4(ln-1,sn);
        X4(ln-1,5)= X4(ln-1,5)+ Yo4(ln-1, sn-2)*dU4(ln-1,sn);
        X4(ln-1,6)= X4(ln-1,6)+ Yw4(ln-1, sn-2)*dU4(ln-1,sn);
        X4(ln-1,7)= X4(ln-1,7)+ Yo4(ln-1, sn-3)*dU4(ln-1,sn);
        X4(ln-1,8)= X4(ln-1,8)+ Yw4(ln-1, sn-3)*dU4(ln-1,sn);
        X4(ln-1,9)= X4(ln-1,9)+ U4(ln-1, sn)*dU4(ln-1,sn);
       
       
        X5(ln-1,1)= X5(ln-1,1)+ Yo5(ln-1, sn)*dU5(ln-1,sn);
        X5(ln-1,2)= X5(ln-1,2)+ Yw5(ln-1, sn)*dU5(ln-1,sn);
        X5(ln-1,3)= X5(ln-1,3)+ Yo5(ln-1, sn-1)*dU5(ln-1,sn);
        X5(ln-1,4)= X5(ln-1,4)+ Yw5(ln-1, sn-1)*dU5(ln-1,sn);
        X5(ln-1,5)= X5(ln-1,5)+ Yo5(ln-1, sn-2)*dU5(ln-1,sn);
        X5(ln-1,6)= X5(ln-1,6)+ Yw5(ln-1, sn-2)*dU5(ln-1,sn);
        X5(ln-1,7)= X5(ln-1,7)+ Yo5(ln-1, sn-3)*dU5(ln-1,sn);
        X5(ln-1,8)= X5(ln-1,8)+ Yw5(ln-1, sn-3)*dU5(ln-1,sn);
        X5(ln-1,9)= X5(ln-1,9)+ U5(ln-1, sn)*dU5(ln-1,sn);
       
       
        X6(ln-1,1)= X6(ln-1,1)+ Yo6(ln-1, sn)*dU6(ln-1,sn);
        X6(ln-1,2)= X6(ln-1,2)+ Yw6(ln-1, sn)*dU6(ln-1,sn);
        X6(ln-1,3)= X6(ln-1,3)+ Yo6(ln-1, sn-1)*dU6(ln-1,sn);
        X6(ln-1,4)= X6(ln-1,4)+ Yw6(ln-1, sn-1)*dU6(ln-1,sn);
        X6(ln-1,5)= X6(ln-1,5)+ Yo6(ln-1, sn-2)*dU6(ln-1,sn);
        X6(ln-1,6)= X6(ln-1,6)+ Yw6(ln-1, sn-2)*dU6(ln-1,sn);
        X6(ln-1,7)= X6(ln-1,7)+ Yo6(ln-1, sn-3)*dU6(ln-1,sn);
        X6(ln-1,8)= X6(ln-1,8)+ Yw6(ln-1, sn-3)*dU6(ln-1,sn);
        X6(ln-1,9)= X6(ln-1,9)+ U6(ln-1, sn)*dU6(ln-1,sn);
        
        
        X7(ln-1,1)= X7(ln-1,1)+ Yo7(ln-1, sn)*dU7(ln-1,sn);
        X7(ln-1,2)= X7(ln-1,2)+ Yw7(ln-1, sn)*dU7(ln-1,sn);
        X7(ln-1,3)= X7(ln-1,3)+ Yo7(ln-1, sn-1)*dU7(ln-1,sn);
        X7(ln-1,4)= X7(ln-1,4)+ Yw7(ln-1, sn-1)*dU7(ln-1,sn);
        X7(ln-1,5)= X7(ln-1,5)+ Yo7(ln-1, sn-2)*dU7(ln-1,sn);
        X7(ln-1,6)= X7(ln-1,6)+ Yw7(ln-1, sn-2)*dU7(ln-1,sn);
        X7(ln-1,7)= X7(ln-1,7)+ Yo7(ln-1, sn-3)*dU7(ln-1,sn);
        X7(ln-1,8)= X7(ln-1,8)+ Yw7(ln-1, sn-3)*dU7(ln-1,sn);
        X7(ln-1,9)= X7(ln-1,9)+ U7(ln-1, sn)*dU7(ln-1,sn);
        
        
        X8(ln-1,1)= X8(ln-1,1)+ Yo8(ln-1, sn)*dU8(ln-1,sn);
        X8(ln-1,2)= X8(ln-1,2)+ Yw8(ln-1, sn)*dU8(ln-1,sn);
        X8(ln-1,3)= X8(ln-1,3)+ Yo8(ln-1, sn-1)*dU8(ln-1,sn);
        X8(ln-1,4)= X8(ln-1,4)+ Yw8(ln-1, sn-1)*dU8(ln-1,sn);
        X8(ln-1,5)= X8(ln-1,5)+ Yo8(ln-1, sn-2)*dU8(ln-1,sn);
        X8(ln-1,6)= X8(ln-1,6)+ Yw8(ln-1, sn-2)*dU8(ln-1,sn);
        X8(ln-1,7)= X8(ln-1,7)+ Yo8(ln-1, sn-3)*dU8(ln-1,sn);
        X8(ln-1,8)= X8(ln-1,8)+ Yw8(ln-1, sn-3)*dU8(ln-1,sn);
        X8(ln-1,9)= X8(ln-1,9)+ U8(ln-1, sn)*dU8(ln-1,sn);
        
        sn=sn+1;
        
    end
    
    ln=ln+1;
end

X=[X X2 X3 X4 X5 X6 X7 X8];
[Theta,BINT,R,RINT,STATS]=regress(Y,X) 

