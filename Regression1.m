%load('controls.mat')
ln=2; nT=50;
dJ=[];
dU1=[]; dU2=[]; dU3=[]; dU4=[]; dU5=[]; dU6=[]; dU7=[]; dU8=[];
TotNPV=NPVTraj(300, :);


while ln<=nT,
    
    dJ=[dJ, TotNPV(ln)-TotNPV(ln-1)];
    dU1=[dU1, U1Traj(:,ln)-U1Traj(:,ln-1)];
    dU2=[dU2, U2Traj(:,ln)-U2Traj(:,ln-1)];
    dU3=[dU3, U3Traj(:,ln)-U3Traj(:,ln-1)];
    dU4=[dU4, U4Traj(:,ln)-U4Traj(:,ln-1)];
    dU5=[dU5, U5Traj(:,ln)-U5Traj(:,ln-1)];
    dU6=[dU6, U6Traj(:,ln)-U6Traj(:,ln-1)];
    dU7=[dU7, U7Traj(:,ln)-U7Traj(:,ln-1)];
    dU8=[dU8, U8Traj(:,ln)-U8Traj(:,ln-1)];

    
    ln=ln+1;
end
    dJ=dJ';
    dU1=dU1';
    dU2=dU2';
    dU3=dU3';
    dU4=dU4';
    dU5=dU5';
    dU6=dU6';
    dU7=dU7';
    dU8=dU8';
