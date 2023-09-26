
%% Define geometry and rock parameters
% 

nx = 20; ny = 20; nz = 5;
G = cartGrid([nx, ny, nz], [5*nx, ny*4, nz*2]);
G = computeGeometry(G);

%load permeability fields
load('perDistr')
%[K, L] = logNormLayers([nx, ny, nz], [200 500 350 700 250], 'sigma', 1);
rock.perm = K*milli*darcy;
rock.poro  = repmat(0.3, [G.cells.num, 1]);

%% Define the two-phase fluid model
%
fluid = initSimpleFluid('mu' , [   1,  10] .* centi*poise     , ...
                        'rho', [1000, 700] .* kilogram/meter^3, ...
                        'n'  , [   2,   2]);

%%
%
s=linspace(0,1,20)'; kr=fluid.relperm(s);

%% Initialize and construct linear system
S  = computeMimeticIP(G, rock, 'Verbose', true);

%% Introduce wells
% <html>
% We will include four wells, all rate-controlled vertical wells. Wells are described
% using a Peacemann model, giving an extra set of equations that need to be
% assembled, see <a href="../../1ph/html/simpleWellExample.html#3"> "Using
% Peacemann well models"</a> for more details.
% </html>
% load control
%u1 = 0.3848/day;   % use OC initial rate
%u1 = 10.5787/day; u2 = 28.3743/day; % use BM initial rate
% use material bal for initial rate
totTime = 730*day;
nInj = 8;
totVol = sum(poreVolume(G, rock));
u1 = (1/nInj)*totVol/totTime;
u2 = u1;
u3 = u1;
u4 = u1;
u5 = u1;
u6 = u1;
u7 = u1;
u8 = u1;

%%
radius     = .1;
W = [];
range = 1:nx*ny:nx*ny*nz;
cellsWell1 = range(1);
W = addWell(W, G, rock, cellsWell1, 'Type', 'rate', ...
            'Val', u1, 'Radius', radius, 'comp_i', [1,0], 'sign', 1,'name', 'I1');
disp('Well #1: '); display(W(1));

%cellsWell2 =  nx*ny+1;
cellsWell2 =  range(2);

W = addWell(W, G, rock, cellsWell2, 'Type', 'rate', ...
            'Val', u2, 'Radius', radius, 'comp_i', [1,0], 'sign', 1,'name', 'I2');
disp('Well #2: '); display(W(2));

%cellsWell3 =  2*nx*ny+1;
cellsWell3 =  range(3);
W = addWell(W, G, rock, cellsWell3, 'Type', 'rate', ...
            'Val', u3, 'Radius', radius, 'comp_i', [1,0], 'sign', 1,'name', 'I3');
disp('Well #3: '); display(W(3));

%cellsWell4 =  3*nx*ny+1;
cellsWell4 =  range(4);
W = addWell(W, G, rock, cellsWell4, 'Type', 'rate', ...
            'Val', u4, 'Radius', radius, 'comp_i', [1,0], 'sign', 1,'name', 'I4');
disp('Well #4: '); display(W(4));

range2 = 20:nx*ny:nx*ny*nz;
cellsWell5 =  range2(1);
W = addWell(W, G, rock, cellsWell5, 'Type', 'rate', ...
            'Val', u5, 'Radius', radius, 'comp_i', [1,0], 'sign', 1,'name', 'I5');
disp('Well #5: '); display(W(5));

%cellsWell6 =  (nx*ny)+20;
cellsWell6 =  range2(2);
W = addWell(W, G, rock, cellsWell6, 'Type', 'rate', ...
            'Val', u6, 'Radius', radius, 'comp_i', [1,0], 'sign', 1,'name', 'I6');
disp('Well #6: '); display(W(6));

%cellsWell7 =  (2*nx*ny)+20;
cellsWell7 =  range2(3);
W = addWell(W, G, rock, cellsWell7, 'Type', 'rate', ...
            'Val', u7, 'Radius', radius, 'comp_i', [1,0], 'sign', 1,'name', 'I7');
disp('Well #7: '); display(W(7));

%cellsWell8 =  (3*nx*ny)+20;
cellsWell8 =  range2(4);
W = addWell(W, G, rock, cellsWell8, 'Type', 'rate', ...
            'Val', u8, 'Radius', radius, 'comp_i', [1,0], 'sign', 1,'name', 'I8');
disp('Well #8: '); display(W(8));




%} Addition of Production wells
nProd = 8;

range3 = 400:nx*ny:nx*ny*nz;
cellsWell9 =  range3(1);
W = addWell(W, G, rock, cellsWell9, 'Type', 'rate', ...
            'Val', -u1, 'Radius', radius, 'comp_i', [1,1], 'sign', -1,'name', 'P9');
disp('Well #9: '); display(W(9));

%cellsWell10 =  2*nx*ny;
cellsWell10 = range3(2);
W = addWell(W, G, rock, cellsWell10, 'Type', 'rate', ...
            'Val', -u2, 'Radius', radius, 'comp_i', [1,1], 'sign', -1,'name', 'P10');
disp('Well #10: '); display(W(10));

%cellsWell11 =  3*nx*ny;
cellsWell11 = range3(3);
W = addWell(W, G, rock, cellsWell11, 'Type', 'rate', ...
            'Val', -u3, 'Radius', radius, 'comp_i', [1,1], 'sign', -1,'name', 'P11');
disp('Well #11: '); display(W(11));

%cellsWell12 =  4*nx*ny;
cellsWell12 = range3(4);
W = addWell(W, G, rock, cellsWell12, 'Type', 'rate', ...
            'Val', -u4, 'Radius', radius, 'comp_i', [1,1], 'sign', -1,'name', 'P12');
disp('Well #12: '); display(W(12));

range4 = 381:nx*ny:nx*ny*nz;
cellsWell13 =  range4(1);
W = addWell(W, G, rock, cellsWell13, 'Type', 'rate', ...
            'Val', -u5, 'Radius', radius, 'comp_i', [1,1], 'sign', -1,'name', 'P13');
disp('Well #13: '); display(W(13));

%cellsWell14 =  (39*nx)+1;
cellsWell14 = range4(2);
W = addWell(W, G, rock, cellsWell14, 'Type', 'rate', ...
            'Val', -u6, 'Radius', radius, 'comp_i', [1,1], 'sign', -1,'name', 'P14');
disp('Well #14: '); display(W(14));

%cellsWell15 =  (59*nx)+1;
cellsWell15 = range4(3);
W = addWell(W, G, rock, cellsWell15, 'Type', 'rate', ...
            'Val', -u7, 'Radius', radius, 'comp_i', [1,1], 'sign', -1,'name', 'P15');
disp('Well #15: '); display(W(15));

%cellsWell16 =  (79*nx)+1;
cellsWell16 = range4(4);
W = addWell(W, G, rock, cellsWell16, 'Type', 'rate', ...
            'Val', -u8, 'Radius', radius, 'comp_i', [1,1], 'sign', -1,'name', 'P16');
disp('Well #16: '); display(W(16));

%}
% To check if the wells are placed as we wanted them, we plot them
plotGrid(G, 'FaceColor', 'none'); view(3);
[ht, htxt, hs] = plotWell(G, W, 'radius', 0.1, 'height', 2);
set(htxt, 'FontSize', 16);


%%
%
rSol = initState(G, W, 0, [0, 1]);

%% Solve initial pressure in reservoir
% Solve linear system construced from S and W to obtain solution for flow
% and pressure in the reservoir and the wells.
gravity off
rSol = solveIncompFlow(rSol, G, S, fluid, 'wells', W);


%% Transport loop
% 
T      = 730*day();
dT     = T/5;
pv     = poreVolume(G,rock);

%%
%
rISol = rSol;


mu=fluid.properties();


opr = @(m,q)sum(m(:,2).*q./sum(m,2));
wpr = @(m,q)sum(m(:,1).*q./sum(m,2));
wir = @(m,q)sum(m(:,1).*q./sum(m,2));

%Define prices in $/m^3 and discount factor
op=100/0.1589873; %100$/bbl
wp=10/0.1589873;
wi=10/0.1589873;
b=0;

%W(1).val=controls.well(1).values(1); W(2).val=controls.well(2).values(1); 
%W(3).val=-W(2).val; W(4).val=-W(1).val;     
rISol = initState(G, W, 0, [0, 1]);   
gravity off;
rISol = solveIncompFlow(rISol, G, S, fluid, 'wells', W);


%% Start the main loop
t  = 0; plotNo = 1; hi = 'Implicit: '; he = 'Explicit: ';
po9socC1 = []; po10socC1 = []; po11socC1 = []; po12socC1 = []; po13socC1 = []; po14socC1 = []; po15socC1 = [];
po16socC1 = []; 
pw9socC1 = []; Yw1 = []; pw10socC1 = []; pw11socC1 = []; pw12socC1 = []; pw13socC1 = []; pw14socC1 = []; pw15socC1 = [];
pw16socC1 = [];
usoc1C1 = []; usoc2C1 = []; usoc3C1 = []; usoc4C1 = []; usoc5C1 = []; usoc6C1 = []; usoc7C1 = []; usoc8C1 = []; 
NPV1=0; NPVsocC1=[]; l = 0;

%load('mytheta')
while t < T,
   
   l = l + 1; 
   
   
   rISol = implicitTransport(rISol, G, dT, rock, fluid, 'wells', W);
   W(1).val = u1; W(2).val = u2; W(3).val = u3; W(4).val = u4; W(5).val = u5;   W(6).val = u6; W(7).val = u7; W(8).val = u8;
W(9).val = -u8; W(10).val = -u7; W(11).val = -u6; W(12).val = -u5; W(13).val = -u4; W(14).val = -u3; W(15).val = -u2; 
W(16).val = -u1;
   
    mSol1=bsxfun(@rdivide, fluid.relperm(rISol.s(W(1).cells,:)), mu);
   mSol2=bsxfun(@rdivide, fluid.relperm(rISol.s(W(2).cells,:)), mu);
   mSol3=bsxfun(@rdivide, fluid.relperm(rISol.s(W(3).cells,:)), mu);
   mSol4=bsxfun(@rdivide, fluid.relperm(rISol.s(W(4).cells,:)), mu);
   mSol5=bsxfun(@rdivide, fluid.relperm(rISol.s(W(5).cells,:)), mu);
   mSol6 =bsxfun(@rdivide, fluid.relperm(rISol.s(W(6).cells,:)), mu);
   mSol7=bsxfun(@rdivide, fluid.relperm(rISol.s(W(7).cells,:)), mu);
   mSol8=bsxfun(@rdivide, fluid.relperm(rISol.s(W(8).cells,:)), mu);
   mSol9=bsxfun(@rdivide, fluid.relperm(rISol.s(W(9).cells,:)), mu);
   mSol10=bsxfun(@rdivide, fluid.relperm(rISol.s(W(10).cells,:)), mu);
   mSol11=bsxfun(@rdivide, fluid.relperm(rISol.s(W(11).cells,:)), mu);
   mSol12=bsxfun(@rdivide, fluid.relperm(rISol.s(W(12).cells,:)), mu);
   mSol13=bsxfun(@rdivide, fluid.relperm(rISol.s(W(13).cells,:)), mu);
   mSol14=bsxfun(@rdivide, fluid.relperm(rISol.s(W(14).cells,:)), mu);
   mSol15=bsxfun(@rdivide, fluid.relperm(rISol.s(W(15).cells,:)), mu);
   mSol16=bsxfun(@rdivide, fluid.relperm(rISol.s(W(16).cells,:)), mu);
   
   OilPr9 = opr(mSol9, -rISol.wellSol(9).flux);              
   OilPr10 = opr(mSol10, -rISol.wellSol(10).flux);
   OilPr11 = opr(mSol11, -rISol.wellSol(11).flux);            
   OilPr12 = opr(mSol12, -rISol.wellSol(12).flux);
   OilPr13 = opr(mSol13, -rISol.wellSol(13).flux);
   OilPr14 = opr(mSol14, -rISol.wellSol(14).flux);
   OilPr15 = opr(mSol15, -rISol.wellSol(15).flux);
   OilPr16 = opr(mSol16, -rISol.wellSol(16).flux);
   
   WatPr9 = wpr(mSol9, -rISol.wellSol(9).flux);            
   WatPr10 = wpr(mSol10, -rISol.wellSol(10).flux);
   WatPr11 = wpr(mSol11, -rISol.wellSol(11).flux);
   WatPr12 = wpr(mSol12, -rISol.wellSol(12).flux);
   WatPr13 = wpr(mSol13, -rISol.wellSol(13).flux);
   WatPr14 = wpr(mSol14, -rISol.wellSol(14).flux);
   WatPr15 = wpr(mSol15, -rISol.wellSol(15).flux);
   WatPr16 = wpr(mSol16, -rISol.wellSol(16).flux);
   
   WatIn1 = wir(mSol1, rISol.wellSol(1).flux);           
   WatIn2 = wir(mSol2, rISol.wellSol(2).flux);
   WatIn3 = wir(mSol3, rISol.wellSol(3).flux);
   WatIn4 = wir(mSol4, rISol.wellSol(4).flux);
   WatIn5 = wir(mSol5, rISol.wellSol(5).flux);
   WatIn6 = wir(mSol6, rISol.wellSol(6).flux);
   WatIn7 = wir(mSol7, rISol.wellSol(7).flux);
   WatIn8 = wir(mSol8, rISol.wellSol(8).flux);
   
   po9socC1 = [po9socC1; OilPr9];                 %#ok
   po10socC1 = [po10socC1; OilPr10];
   po11socC1 = [po11socC1; OilPr11];
   po12socC1 = [po12socC1; OilPr12];
   po13socC1 = [po13socC1; OilPr13];
   po14socC1 = [po14socC1; OilPr14];
   po15socC1 = [po15socC1; OilPr15];
   po16socC1 = [po16socC1; OilPr16];
   
   pw9socC1 = [pw9socC1; WatPr9];                 %#ok
   pw10socC1 = [pw10socC1; WatPr10];
   pw11socC1 = [pw11socC1; WatPr11];
   pw12socC1 = [pw12socC1; WatPr12];
   pw13socC1 = [pw13socC1; WatPr13];
   pw14socC1 = [pw14socC1; WatPr14];
   pw15socC1 = [pw15socC1; WatPr15];
   pw16socC1 = [pw16socC1; WatPr16];
   
   %uw1 = [uw1; WatIn1];                          %#ok
   %uw2 = [uw2; WatIn2];
   usoc1C1 = [usoc1C1; W(1).val];                 %#ok
   usoc2C1 = [usoc2C1; W(2).val];
    usoc3C1 = [usoc3C1; W(3).val];
   usoc4C1 = [usoc4C1; W(4).val];
  usoc5C1 = [usoc5C1; W(5).val];
   usoc6C1 = [usoc6C1; W(6).val];
  usoc7C1 = [usoc7C1; W(7).val];
  usoc8C1 = [usoc8C1; W(8).val];
   
   NPV1=NPV1+dT*(-(WatIn1+WatIn2+WatIn3+WatIn4+WatIn5+WatIn6+WatIn7+WatIn8)*wi-(WatPr9+WatPr10+WatPr11+WatPr12+WatPr13+WatPr14+WatPr15+WatPr16)*wp+(OilPr9+OilPr10+OilPr11+OilPr12+OilPr13+OilPr14+OilPr15+OilPr16)*op)/(1+b)^t;
   NPVsocC1=[NPVsocC1; NPV1];
   
   % Update solution of pressure equation.
   rISol = solveIncompFlow(rISol, G, S, fluid, 'wells', W);
   
   %Define next value of the CVs, using the feedback control law if there
   %are all ready two available past histories
   %
   if l > 3 % number of history check
   
ucv1 = -(Theta(9))^-1*(Theta(1)*po16socC1(l)+Theta(2)*pw16socC1(l)+Theta(3)*po16socC1(l-1)+Theta(4)*pw16socC1(l-1)+Theta(5)*po16socC1(l-2)+Theta(6)*pw16socC1(l-2)+Theta(7)*po16socC1(l-3)+Theta(8)*pw16socC1(l-3)); 
ucv2 = -(Theta(18))^-1*(Theta(10)*po15socC1(l)+Theta(11)*pw15socC1(l)+Theta(12)*po15socC1(l-1)+Theta(13)*pw15socC1(l-1)+Theta(14)*po15socC1(l-2)+Theta(15)*pw15socC1(l-2)+ Theta(16)*po16socC1(l-3)+Theta(17)*pw16socC1(l-3)); 
ucv3 = -(Theta(27))^-1*(Theta(19)*po14socC1(l)+Theta(20)*pw14socC1(l)+Theta(21)*po14socC1(l-1)+Theta(22)*pw14socC1(l-1)+Theta(23)*po14socC1(l-2)+Theta(24)*pw14socC1(l-2)+ Theta(25)*po16socC1(l-3)+Theta(26)*pw16socC1(l-3));  
ucv4 = -(Theta(36))^-1*(Theta(28)*po13socC1(l)+Theta(29)*pw13socC1(l)+Theta(30)*po13socC1(l-1)+Theta(31)*pw13socC1(l-1)+Theta(32)*po13socC1(l-2)+Theta(33)*pw13socC1(l-2)+ Theta(34)*po16socC1(l-3)+Theta(35)*pw16socC1(l-3));
ucv5 = -(Theta(45))^-1*(Theta(37)*po12socC1(l)+Theta(38)*pw12socC1(l)+Theta(39)*po12socC1(l-1)+Theta(40)*pw12socC1(l-1)+Theta(41)*po12socC1(l-2)+Theta(42)*pw12socC1(l-2)+ Theta(43)*po16socC1(l-3)+Theta(44)*pw16socC1(l-3));
ucv6 = -(Theta(54))^-1*(Theta(46)*po11socC1(l)+Theta(47)*pw11socC1(l)+Theta(48)*po11socC1(l-1)+Theta(49)*pw11socC1(l-1)+Theta(50)*po11socC1(l-2)+Theta(51)*pw11socC1(l-2)+ Theta(52)*po16socC1(l-3)+Theta(53)*pw16socC1(l-3));  
ucv7 = -(Theta(63))^-1*(Theta(55)*po10socC1(l)+Theta(56)*pw10socC1(l)+Theta(57)*po10socC1(l-1)+Theta(58)*pw10socC1(l-1)+Theta(59)*po10socC1(l-2)+Theta(60)*pw10socC1(l-2)+ Theta(61)*po16socC1(l-3)+Theta(62)*pw16socC1(l-3));   
ucv8 = -(Theta(72))^-1*(Theta(64)*po9socC1(l)+Theta(65)*pw9socC1(l)+Theta(66)*po9socC1(l-1)+Theta(67)*pw9socC1(l-1)+Theta(68)*po9socC1(l-2)+Theta(69)*pw9socC1(l-2)+ Theta(70)*po16socC1(l-3)+Theta(71)*pw16socC1(l-3));  
   
   u1 = ucv1;
   u2 = ucv2;
   u3 = ucv3;
   u4 = ucv4;
   u5 = ucv5;
   u6 = ucv6;
   u7 = ucv7;
   u8 = ucv8;
  
   W(1).val = u1; W(2).val = u2; W(3).val = u3; W(4).val = u4; W(5).val = u5;   W(6).val = u6; W(7).val = u7; W(8).val = u8;
W(9).val = -u8; W(10).val = -u7; W(11).val = -u6; W(12).val = -u5; W(13).val = -u4; W(14).val = -u3; W(15).val = -u2; 
W(16).val = -u1;
   
   l
  gradient1 = Theta(1)*po16socC1(l)+Theta(2)*pw16socC1(l)+Theta(3)*po16socC1(l-1)+Theta(4)*pw16socC1(l-1)+Theta(5)*po16socC1(l-2)+Theta(6)*pw16socC1(l-2)+ Theta(7)*po16socC1(l-3)+Theta(8)*pw16socC1(l-3)+ Theta(9)*ucv1
   gradient2 = Theta(10)*po15socC1(l)+Theta(11)*pw15socC1(l)+Theta(12)*po15socC1(l-1)+Theta(13)*pw15socC1(l-1)+Theta(14)*po15socC1(l-2)+Theta(15)*pw15socC1(l-2)+ Theta(16)*po16socC1(l-3)+Theta(17)*pw16socC1(l-3)+ Theta(18)*ucv2
   gradient3 = Theta(19)*po14socC1(l)+Theta(20)*pw14socC1(l)+Theta(21)*po14socC1(l-1)+Theta(22)*pw14socC1(l-1)+Theta(23)*po14socC1(l-2)+Theta(24)*pw14socC1(l-2)+ Theta(25)*po16socC1(l-3)+Theta(26)*pw16socC1(l-3)+ Theta(27)*ucv3
   gradient4 = Theta(28)*po13socC1(l)+Theta(29)*pw13socC1(l)+Theta(30)*po13socC1(l-1)+Theta(31)*pw13socC1(l-1)+Theta(32)*po13socC1(l-2)+Theta(33)*pw13socC1(l-2)+Theta(34)*po16socC1(l-3)+Theta(35)*pw16socC1(l-3)+ Theta(36)*ucv4
   gradient5 = Theta(37)*po12socC1(l)+Theta(38)*pw12socC1(l)+Theta(39)*po12socC1(l-1)+Theta(40)*pw12socC1(l-1)+Theta(41)*po12socC1(l-2)+Theta(42)*pw12socC1(l-2)+Theta(43)*po16socC1(l-3)+Theta(44)*pw16socC1(l-3)+ Theta(45)*ucv5
   gradient6 = Theta(46)*po11socC1(l)+Theta(47)*pw11socC1(l)+Theta(48)*po11socC1(l-1)+Theta(49)*pw11socC1(l-1)+Theta(50)*po11socC1(l-2)+Theta(51)*pw11socC1(l-2)+Theta(52)*po16socC1(l-3)+Theta(53)*pw16socC1(l-3)+ Theta(54)*ucv6 
   gradient7 = Theta(55)*po10socC1(l)+Theta(56)*pw10socC1(l)+Theta(57)*po10socC1(l-1)+Theta(58)*pw10socC1(l-1)+Theta(59)*po10socC1(l-2)+Theta(60)*pw10socC1(l-2)+Theta(61)*po16socC1(l-3)+Theta(62)*pw16socC1(l-3)+ Theta(63)*ucv7
   gradient8 = Theta(64)*po9socC1(l)+Theta(65)*pw9socC1(l)+Theta(66)*po9socC1(l-1)+Theta(67)*pw9socC1(l-1)+Theta(68)*po9socC1(l-2)+Theta(69)*pw9socC1(l-2)+Theta(70)*po16socC1(l-3)+Theta(71)*pw16socC1(l-3)+ Theta(72)*ucv8
   
   Un=[ucv1 ucv2 ucv3 ucv4 ucv5 ucv6 ucv7 ucv8]

   end
   %}

   t = t + dT;
end


%%
% Plot the water and oil rates
n = size(po9socC1,1);

figure
subplot(2,4,1),
   plot(1:n,3600*24*po9socC1(:,1))
   title('Oil production in well P');
subplot(2,4,2),
   plot(1:n,3600*24*po10socC1(:,1))
   title('Oil production in well P1');
subplot(2,4,3),
   plot(1:n,3600*24*po11socC1(:,1))
   title('Oil production in well P1');
subplot(2,4,4),
   plot(1:n,3600*24*po12socC1(:,1))
   title('Oil production in well P1');
subplot(2,4,5),
   plot(1:n,3600*24*po13socC1(:,1))
   title('Oil production in well P1');
subplot(2,4,6),
   plot(1:n,3600*24*po14socC1(:,1))
   title('Oil production in well P1');
subplot(2,4,7),
   plot(1:n,3600*24*po15socC1(:,1))
   title('Oil production in well P1');
subplot(2,4,8),
   plot(1:n,3600*24*po16socC1(:,1))
   title('Oil production in well P1');
   
   figure
subplot(2,4,1),
   plot(1:n,3600*24*pw9socC1(:,1))
   title('Water production in well P');
subplot(2,4,2),
   plot(1:n,3600*24*pw10socC1(:,1))
   title('Water production in well P1');
subplot(2,4,3),
   plot(1:n,3600*24*pw11socC1(:,1))
   title('Water production in well P1');
subplot(2,4,4),
   plot(1:n,3600*24*pw12socC1(:,1))
   title('Water production in well P1');
subplot(2,4,5),
   plot(1:n,3600*24*pw13socC1(:,1))
   title('Water production in well P1');
subplot(2,4,6),
   plot(1:n,3600*24*pw14socC1(:,1))
   title('Water production in well P1');
subplot(2,4,7),
   plot(1:n,3600*24*pw15socC1(:,1))
   title('Water production in well P1');
subplot(2,4,8),
   plot(1:n,3600*24*pw16socC1(:,1))
   title('Water production in well P1');
   
figure
subplot(2,4,1),
   plot(1:n,3600*24*usoc1C1(:,1))
   title('Water injection in well I');
subplot(2,4,2),
   plot(1:n,3600*24*usoc2C1(:,1))
   title('Water injection in well I2');
subplot(2,4,3),
   plot(1:n,3600*24*usoc3C1(:,1))
   title('Water injection in well I2');
subplot(2,4,4),
   plot(1:n,3600*24*usoc4C1(:,1))
   title('Water injection in well I2');
subplot(2,4,5),
   plot(1:n,3600*24*usoc5C1(:,1))
   title('Water injection in well I2');
subplot(2,4,6),
   plot(1:n,3600*24*usoc6C1(:,1))
   title('Water injection in well I2');
subplot(2,4,7),
   plot(1:n,3600*24*usoc7C1(:,1))
   title('Water injection in well I2');
subplot(2,4,8),
   plot(1:n,3600*24*usoc8C1(:,1))
   title('Water injection in well I2');
   
%Plot the NPV

figure
   plot(1:n,NPVsocC1(:,1))
   title('Net Present Value');
   
% total production
oilsoc9 = po9socC1*60*60*24;
    oilsoc10 = po10socC1*60*60*24;
    oilsoc11 = po11socC1*60*60*24;
    oilsoc12 = po12socC1*60*60*24;
    oilsoc13 = po13socC1*60*60*24;
    oilsoc14 = po14socC1*60*60*24;
    oilsoc15 = po15socC1*60*60*24;
    oilsoc16 = po16socC1*60*60*24;
   
   watersoc9 = pw9socC1*60*60*24;
   watersoc10 = pw10socC1*60*60*24;
   watersoc11 = pw11socC1*60*60*24;
   watersoc12 = pw12socC1*60*60*24;
   watersoc13 = pw13socC1*60*60*24;
   watersoc14 = pw14socC1*60*60*24;
   watersoc15 = pw15socC1*60*60*24;
   watersoc16 = pw16socC1*60*60*24;
   
  injsoc1 = usoc1C1*60*60*24;
  injsoc2 = usoc2C1*60*60*24;
  injsoc3 = usoc3C1*60*60*24;
  injsoc4 = usoc4C1*60*60*24;
  injsoc5 = usoc5C1*60*60*24;
  injsoc6 = usoc6C1*60*60*24;
  injsoc7 = usoc7C1*60*60*24;
  injsoc8 = usoc8C1*60*60*24;
  
  
Tvec = (dT : dT : T)';
oilT1 = cumtrapz ([0; Tvec], [zeros([1, 1]); po9socC1]);
oilT2 = cumtrapz ([0; Tvec], [zeros([1, 1]); po10socC1]);
oilT3 = cumtrapz ([0; Tvec], [zeros([1, 1]); po11socC1]);
oilT4 = cumtrapz ([0; Tvec], [zeros([1, 1]); po12socC1]);
oilT5 = cumtrapz ([0; Tvec], [zeros([1, 1]); po13socC1]);
oilT6 = cumtrapz ([0; Tvec], [zeros([1, 1]); po14socC1]);
oilT7 = cumtrapz ([0; Tvec], [zeros([1, 1]); po15socC1]);
oilT8 = cumtrapz ([0; Tvec], [zeros([1, 1]); po16socC1]);
wat1 = cumtrapz ([0; Tvec], [zeros([1, 1]); pw9socC1]);
wat2 = cumtrapz ([0; Tvec], [zeros([1, 1]); pw10socC1]);
wat3 = cumtrapz ([0; Tvec], [zeros([1, 1]); pw11socC1]);
wat4 = cumtrapz ([0; Tvec], [zeros([1, 1]); pw12socC1]);
wat5 = cumtrapz ([0; Tvec], [zeros([1, 1]); pw13socC1]);
wat6 = cumtrapz ([0; Tvec], [zeros([1, 1]); pw14socC1]);
wat7 = cumtrapz ([0; Tvec], [zeros([1, 1]); pw15socC1]);
wat8 = cumtrapz ([0; Tvec], [zeros([1, 1]); pw16socC1]);

totaloil = oilT1+oilT2+oilT3+oilT4+oilT5+oilT6+oilT7+oilT8;
totalwater = wat1+wat2+wat3+wat4+wat5+wat6+wat7+wat8;
%%
displayEndOfDemoMessage(mfilename)
