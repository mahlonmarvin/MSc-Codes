 
%load('controls.mat')
%% Define geometry and rock parameters
% Construct a Cartesian grid of size 20-by-20-by-5 cells, where each cell
% has dimension 1-by-1-by-1. Set the permeability $K$ to be homogeneous,
% isotropic and equal 100 mD and the porosity to be equal to 0.3.
nx = 20; ny = 20; nz = 5;
G = cartGrid([nx, ny, nz]);
G = computeGeometry(G);
rock.perm  = repmat(100*milli*darcy, [G.cells.num, 1]);
rock.poro  = repmat(0.3, [G.cells.num, 1]);

%% Define the two-phase fluid model
% The <matlab:help('initSimpleFluid') two-phase fluid model> has default values:
%
% * densities: [rho_w, rho_o] = [1000 700] kg/m^3
% * viscosities: [mu_w, mu_o] = [1 10] cP.
fluid = initSimpleFluid('mu' , [   1,  10] .* centi*poise     , ...
                        'rho', [1000, 700] .* kilogram/meter^3, ...
                        'n'  , [   2,   2]);

%%
% The fluid model represented by the <matlab:help('fluid') fluid structure>
% is the two-phase incompressible counterpart to the fluid model of the
% Black Oil <matlab:help('pvt') 'pvt'> function.
%
figure
s=linspace(0,1,20)'; kr=fluid.relperm(s);
plot(s, kr(:,1), 'b', s, kr(:,2), 'r');
title('Relative permeability curves')
legend('Water','Oil','Location','Best')


%% Initialize and construct linear system
% Initialize solution structure with reservoir pressure equal 0 and initial
% water saturation equal 0.0 (reservoir is filled with oil). Compute the
% mimetic inner product from input grid and rock properties.
S  = computeMimeticIP(G, rock, 'Verbose', true);

%% Introduce wells
% <html>
% We will include four wells, all rate-controlled vertical wells. Wells are described
% using a Peacemann model, giving an extra set of equations that need to be
% assembled, see <a href="../../1ph/html/simpleWellExample.html#3"> "Using
% Peacemann well models"</a> for more details.
% </html>

u1=controls.well(1).values(1); u2=controls.well(2).values(1);
u3=controls.well(3).values(1); u4=controls.well(4).values(1);
u5=controls.well(5).values(1); u6=controls.well(6).values(1);
u7=controls.well(7).values(1); u8=controls.well(8).values(1);


%u1 = 2.8/day; u2 = 2.8/day;



%% Five injection wells
%nInj = 2;
nInj = 8;
radius     = .1;
W = [];
range = 1:nx*ny:nx*ny*nz;
cellsWell1 = range(1)
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
cellsWell5 =  range2(1)
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
cellsWell9 =  range3(1)
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
cellsWell15 = range4(3)
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
figure
plotGrid(G, 'FaceColor', 'none'); view(3);
[ht, htxt, hs] = plotWell(G, W, 'radius', 0.1, 'height', 2);
set(htxt, 'FontSize', 16);
%
%see me here
%%
% Once the wells are added, we can generate the components of the linear
% system corresponding to the four wells and initialize the solution
% structure (with correct bhp)
%
rSol = initState(G, W, 0, [0, 1]);

%% Solve initial pressure in reservoir
% Solve linear system construced from S and W to obtain solution for flow
% and pressure in the reservoir and the wells.
gravity off
rSol = solveIncompFlow(rSol, G, S, fluid, 'wells', W);

%%
% Report initial state of reservoir
%subplot(2,1,1), cla
figure 
plotCellData(G, convertTo(rSol.pressure(1:G.cells.num), barsa));
   title('Initial pressure'), view(3)

%subplot(2,1,2), cla
figure 
cellNo = rldecode(1:G.cells.num, diff(G.cells.facePos), 2) .';
   plotCellData(G, accumarray(cellNo, ...
      abs(convertTo(faceFlux2cellFlux(G, rSol.flux), meter^3/day))));
   title('Initial flux intensity'), view(3)

%% Transport loop
% We solve the two-phase system using a sequential splitting in which the
% pressure and fluxes are computed by solving the flow equation and then
% held fixed as the saturation is advanced according to the transport
% equation. This procedure is repeated for a given number of time steps
% (here we use 15 equally spaced time steps). The error introduced by this
% splitting of flow and transport can be reduced by iterating each time
% step until e.g., the residual is below a certain user-prescribed
% threshold (this is not done herein).
T      = 730*day();
nS = 300;
dT     = T/nS;
pv     = poreVolume(G,rock);

%%
% The transport equation will be solved by the single-point upstream method
% with either explicit or implicit time discretizations. Both schemes may
% use internal time steps to obtain a stable discretization. To represent
% the two solutions, we create new solution objects to be used by the
% solver with implicit transport step.
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

%Loop for 500 trajectories
ln=1; nT=50; pert=0;
OilTraj1=zeros(nS, nT); OilTraj2=zeros(nS, nT);
OilTraj3=zeros(nS, nT); OilTraj4=zeros(nS, nT);
OilTraj5=zeros(nS, nT); OilTraj6=zeros(nS, nT);
OilTraj7=zeros(nS, nT); OilTraj8=zeros(nS, nT);

WProTraj1=zeros(nS, nT); WProTraj2=zeros(nS, nT);
WProTraj3=zeros(nS, nT); WProTraj4=zeros(nS, nT);
WProTraj5=zeros(nS, nT); WProTraj6=zeros(nS, nT);
WProTraj7=zeros(nS, nT); WProTraj8=zeros(nS, nT); 

WInjTraj1=zeros(nS, nT); WInjTraj2=zeros(nS, nT);
WInjTraj3=zeros(nS, nT); WInjTraj4=zeros(nS, nT);
WInjTraj5=zeros(nS, nT); WInjTraj6=zeros(nS, nT);
WInjTraj7=zeros(nS, nT); WInjTraj8=zeros(nS, nT);

U1Traj=zeros(nS, nT); U2Traj=zeros(nS, nT);
U3Traj=zeros(nS, nT); U4Traj=zeros(nS, nT); 
U5Traj=zeros(nS, nT); U6Traj=zeros(nS, nT); 
U7Traj=zeros(nS, nT); U8Traj=zeros(nS, nT); 

NPVTraj=zeros(nS,nT);
%%
while ln<=nT,
    
    if ln>3,        
pert1=(rand(1)/100+0.0003)/day;
pert2=(rand(1)/100+0.0003)/day;
pert3=(rand(1)/100+0.0003)/day;
pert4=(rand(1)/100+0.0003)/day;
pert5=(rand(1)/100+0.0003)/day;
pert6=(rand(1)/100+0.0003)/day;
pert7=(rand(1)/100+0.0003)/day;
pert8=(rand(1)/100+0.0003)/day;

W(1).val=controls.well(1).values(1)+pert1; W(2).val=controls.well(2).values(1)+pert2; 
W(3).val=controls.well(3).values(1)+pert3; W(4).val=controls.well(4).values(1)+pert4;
W(5).val=controls.well(5).values(1)+pert5; W(6).val=controls.well(6).values(1)+pert6;
W(7).val=controls.well(7).values(1)+pert7; W(8).val=controls.well(8).values(1)+pert8;

W(9).val=-W(1).val; W(10).val=-W(2).val;
W(11).val=-W(3).val; W(12).val=-W(4).val;
W(13).val=-W(5).val; W(14).val=-W(6).val;
W(15).val=-W(7).val; W(16).val=-W(8).val;

    else
pert1=0; pert2=0;pert3=0; pert4=0; pert5=0; pert6=0; pert7=0; pert8=0;
W(1).val=controls.well(1).values(1); W(2).val=controls.well(2).values(1); 
W(3).val=controls.well(3).values(1); W(4).val=controls.well(4).values(1);
W(5).val=controls.well(5).values(1); W(6).val=controls.well(6).values(1);
W(7).val=controls.well(7).values(1); W(8).val=controls.well(8).values(1);

W(9).val=-W(1).val; W(10).val=-W(2).val;
W(11).val=-W(3).val; W(12).val=-W(4).val;
W(13).val=-W(5).val; W(14).val=-W(6).val;
W(15).val=-W(7).val; W(16).val=-W(8).val;
    end

rISol = initState(G, W, 0, [0, 1]);   
gravity off;
rISol = solveIncompFlow(rISol, G, S, fluid, 'wells', W);


%% Start the main loop
t  = 0; plotNo = 1; hi = 'Implicit: '; he = 'Explicit: ';
po1 = zeros(nS,1); po2 = zeros(nS,1);
po3 = zeros(nS,1); po4 = zeros(nS,1);
po5 = zeros(nS,1); po6 = zeros(nS,1);
po7 = zeros(nS,1); po8 = zeros(nS,1);

pw1 = zeros(nS,1); pw2 = zeros(nS,1);
pw3 = zeros(nS,1); pw4 = zeros(nS,1);
pw5 = zeros(nS,1); pw6 = zeros(nS,1);
pw7 = zeros(nS,1); pw8 = zeros(nS,1);

uw1 = zeros(nS,1); uw2 = zeros(nS,1);
uw3 = zeros(nS,1); uw4 = zeros(nS,1);
uw5 = zeros(nS,1); uw6 = zeros(nS,1);
uw7 = zeros(nS,1); uw8 = zeros(nS,1);


NPV1=0; NPV= zeros(nS,1); l=1; 

uc1 = zeros(nS,1); uc2 =zeros(nS,1);
uc3 = zeros(nS,1); uc4 =zeros(nS,1);
uc5 = zeros(nS,1); uc6 =zeros(nS,1);
uc7 = zeros(nS,1); uc8 =zeros(nS,1);

while t < T,
   
 %
W(1).val=controls.well(1).values(l)+pert1; W(2).val=controls.well(2).values(l)+pert2; 
W(3).val=controls.well(3).values(l)+pert3; W(4).val=controls.well(4).values(l)+pert4;
W(5).val=controls.well(5).values(l)+pert5; W(6).val=controls.well(6).values(l)+pert6;
W(7).val=controls.well(7).values(l)+pert7; W(8).val=controls.well(8).values(l)+pert8;

W(9).val=-W(1).val; W(10).val=-W(2).val;
W(11).val=-W(3).val; W(12).val=-W(4).val;
W(13).val=-W(5).val; W(14).val=-W(6).val;
W(15).val=-W(7).val; W(16).val=-W(8).val;
    
   rISol = implicitTransport(rISol, G, dT, rock, fluid, 'wells', W);

%%
   % Update solution of pressure equation.
   rISol = solveIncompFlow(rISol, G, S, fluid, 'wells', W);
   
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
  
   % Measure water saturation in production cells in saturation

   OilPr1 = opr(mSol9, -rISol.wellSol(9).flux);
   OilPr2 = opr(mSol10, -rISol.wellSol(10).flux);
   OilPr3 = opr(mSol11, -rISol.wellSol(11).flux);            
   OilPr4 = opr(mSol12, -rISol.wellSol(12).flux);
   OilPr5 = opr(mSol13, -rISol.wellSol(13).flux);
   OilPr6 = opr(mSol14, -rISol.wellSol(14).flux);
   OilPr7 = opr(mSol15, -rISol.wellSol(15).flux);
   OilPr8 = opr(mSol16, -rISol.wellSol(16).flux);
   
   po1(l,:) = OilPr1;
   po2(l,:) = OilPr2;
   po3(l,:) = OilPr3;                 
   po4(l,:) = OilPr4;
   po5(l,:) = OilPr5;
   po6(l,:) = OilPr6;
   po7(l,:) = OilPr7;
   po8(l,:) = OilPr8;
   
   % Measure flux in production cells at every step (implicit solution)

   WatPr1 = wpr(mSol9, -rISol.wellSol(9).flux);
   WatPr2 = wpr(mSol10, -rISol.wellSol(10).flux);
   WatPr3 = wpr(mSol11, -rISol.wellSol(11).flux);           
   WatPr4 = wpr(mSol12, -rISol.wellSol(12).flux);
   WatPr5 = wpr(mSol13, -rISol.wellSol(13).flux);
   WatPr6 = wpr(mSol14, -rISol.wellSol(14).flux);
   WatPr7 = wpr(mSol15, -rISol.wellSol(15).flux);
   WatPr8 = wpr(mSol16, -rISol.wellSol(16).flux);
   
   pw1(l,:) = WatPr1;
   pw2(l,:) = WatPr2;
   pw3(l,:) = WatPr3;                 
   pw4(l,:) = WatPr4;
   pw5(l,:) = WatPr5;
   pw6(l,:) = WatPr6;
   pw7(l,:) = WatPr7;
   pw8(l,:) = WatPr8;
   
   %Measure flux in injection cell at every step 
   
   WatIn1 = wir(mSol1, rISol.wellSol(1).flux);           
   WatIn2 = wir(mSol2, rISol.wellSol(2).flux);
   WatIn3 = wir(mSol3, rISol.wellSol(3).flux);           
   WatIn4 = wir(mSol4, rISol.wellSol(4).flux);
   WatIn5 = wir(mSol5, rISol.wellSol(5).flux);
   WatIn6 = wir(mSol6, rISol.wellSol(6).flux);
   WatIn7 = wir(mSol7, rISol.wellSol(7).flux);
   WatIn8 = wir(mSol8, rISol.wellSol(8).flux);
   
   uw1(l,:) = WatIn1;                 
   uw2(l,:) = WatIn2;
   uw3(l,:) = WatIn3;                 
   uw4(l,:) = WatIn4;
   uw5(l,:) = WatIn5;
   uw6(l,:) = WatIn6;
   uw7(l,:) = WatIn7;
   uw8(l,:) = WatIn8;
   
   uc1(l,:)= W(1).val;
   uc2(l,:)= W(2).val;
   uc3(l,:)= W(3).val;
   uc4(l,:)= W(4).val;
   uc5(l,:)= W(5).val;
   uc6(l,:)= W(6).val;
   uc7(l,:)= W(7).val;
   uc8(l,:)= W(8).val;
   
   %Calculate the NVP value for each step
   
   NPV1=NPV1+dT*(-(WatIn1+WatIn2+WatIn3+WatIn4+WatIn5+WatIn6+WatIn7+WatIn8)*wi-(WatPr1+...
       WatPr2+WatPr3+WatPr4+WatPr5+WatPr6+WatPr7+WatPr8)*wp+(OilPr1+OilPr2+OilPr3+OilPr4+OilPr5+OilPr6+OilPr7+OilPr8)*op)/(1+b)^t;
   %NPV=[NPV; NPV1];
   NPV(l,:)=NPV1;

   % Increase time and continue 
   t = t + dT;
   l=l+1;


end

OilTraj1(:,ln)= po1;
OilTraj2(:,ln)= po2;
OilTraj3(:,ln)= po3;
OilTraj4(:,ln)= po4;
OilTraj5(:,ln)= po5;
OilTraj6(:,ln)= po6;
OilTraj7(:,ln)= po7;
OilTraj8(:,ln)= po8;

WProTraj1(:,ln)=  pw1;
WProTraj2(:,ln) = pw2;
WProTraj3(:,ln)=  pw3;
WProTraj4(:,ln) = pw4;
WProTraj5(:,ln) = pw5;
WProTraj6(:,ln) = pw6;
WProTraj7(:,ln) = pw7;
WProTraj8(:,ln) = pw8;


WInjTraj1(:,ln)= uw1;
WInjTraj2(:,ln)= uw2;
WInjTraj3(:,ln)= uw3;
WInjTraj4(:,ln)= uw4;
WInjTraj5(:,ln)= uw5;
WInjTraj6(:,ln)= uw6;
WInjTraj7(:,ln)= uw7;
WInjTraj8(:,ln)= uw8;


U1Traj(:,ln)= uc1;
U2Traj(:,ln)= uc2;
U3Traj(:,ln)= uc3;
U4Traj(:,ln)= uc4;
U5Traj(:,ln)= uc5;
U6Traj(:,ln)= uc6;
U7Traj(:,ln)= uc7;
U8Traj(:,ln)= uc8;

NPVTraj(:,ln) = NPV;

ln=ln+1   %#ok

end
%%
% Plot the water and oil rates
n = size(po3,1);

figure
subplot(2,4,1),
   plot(1:n,3600*24*po1(:,1))
   title('Oil production in ICV P1');
subplot(2,4,2),
   plot(1:n,3600*24*po2(:,1))
   title('Oil production in ICV P2');
subplot(2,4,3),
   plot(1:n,3600*24*po3(:,1))
   title('Oil production in ICV P3');
subplot(2,4,4),
   plot(1:n,3600*24*po4(:,1))
   title('Oil production in ICV P4');
subplot(2,4,5),
   plot(1:n,3600*24*po5(:,1))
   title('Oil production in ICV P5');
subplot(2,4,6),
   plot(1:n,3600*24*po6(:,1))
   title('Oil production in ICV P6');
subplot(2,4,7),
   plot(1:n,3600*24*po7(:,1))
   title('Oil production in ICV P7');
subplot(2,4,8),
   plot(1:n,3600*24*po8(:,1))
   title('Oil production in ICV P8');
   
   figure
subplot(2,4,1),
   plot(1:n,3600*24*pw1(:,1))
   title('Water production in ICV P1');
subplot(2,4,2),
   plot(1:n,3600*24*pw2(:,1))
   title('Water production in ICV P2');
subplot(2,4,3),
   plot(1:n,3600*24*pw3(:,1))
   title('Water production in ICV P3');
subplot(2,4,4),
   plot(1:n,3600*24*pw4(:,1))
   title('Water production in ICV P4');
subplot(2,4,5),
   plot(1:n,3600*24*pw5(:,1))
   title('Water production in ICV P5');
subplot(2,4,6),
   plot(1:n,3600*24*pw6(:,1))
   title('Water production in ICV P6');
subplot(2,4,7),
   plot(1:n,3600*24*pw7(:,1))
   title('Water production in ICV P7');
subplot(2,4,8),
   plot(1:n,3600*24*pw8(:,1))
   title('Water production in ICV P8');
   
figure
subplot(2,4,1),
   plot(1:n,3600*24*uw1(:,1))
   title('Water injection in ICV I1');
subplot(2,4,2),
   plot(1:n,3600*24*uw2(:,1))
   title('Water injection in ICV I2');
subplot(2,4,3),
   plot(1:n,3600*24*uw3(:,1))
   title('Water injection in ICV I3');
subplot(2,4,4),
   plot(1:n,3600*24*uw4(:,1))
   title('Water injection in ICV I4');
subplot(2,4,5),
   plot(1:n,3600*24*uw5(:,1))
   title('Water injection in ICV I5');
subplot(2,4,6),
   plot(1:n,3600*24*uw6(:,1))
   title('Water injection in ICV I6');
subplot(2,4,7),
   plot(1:n,3600*24*uw7(:,1))
   title('Water injection in ICV I7');
subplot(2,4,8),
   plot(1:n,3600*24*uw8(:,1))
   title('Water injecion in ICV I8');
   
%Plot the NPV

figure
   plot(1:n,NPV(:,1))
   title('Net Present Value');
  %} 


