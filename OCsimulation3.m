
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
                        'n'  , [   1.5,   1.5]);

%%
% The fluid model represented by the <matlab:help('fluid') fluid structure>
% is the two-phase incompressible counterpart to the fluid model of the
% Black Oil <matlab:help('pvt') 'pvt'> function.
%
s=linspace(0,1,20)'; kr=fluid.relperm(s);

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
% Once the wells are added, we can generate the components of the linear
% system corresponding to the two wells and initialize the solution
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
subplot(2,1,1), cla
   plotCellData(G, convertTo(rSol.pressure(1:G.cells.num), barsa));
   title('Initial pressure'), view(3)

subplot(2,1,2), cla
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
dT     = T/5;
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

%% Start the main loop
t  = 0; plotNo = 1; hi = 'Implicit: '; he = 'Explicit: ';
po9ocC1 = []; po10ocC1 = []; po11ocC1 = []; po12ocC1 = []; po13ocC1 = []; po14ocC1 = []; po15ocC1 = []; po16ocC1 = []; 
pw9ocC1 = []; Yw1 = []; pw10ocC1 = []; pw11ocC1 = []; pw12ocC1 = []; pw13ocC1 = []; pw14ocC1 = []; pw15ocC1 = []; pw16ocC1 = [];
uw1 = []; uw2 = []; uw3 = []; uw4 = []; uw5 = []; uw6 = []; uw7 = []; uw8 = [];
NPV1oc=0; NPVocC1=[]; l=1;
uoc1C1=[]; uoc2C1=[]; uoc3C1=[]; uoc4C1=[]; uoc5C1=[]; uoc6C1=[]; uoc7C1=[]; uoc8C1=[];

while t < T,
   l
   u1=controls.well(1).values(1); u2=controls.well(2).values(1);
   u3=controls.well(3).values(1); u4=controls.well(4).values(1);
   u5=controls.well(5).values(1); u6=controls.well(6).values(1);
   u7=controls.well(7).values(1); u8=controls.well(8).values(1);
   
   W(1).val = u1; W(2).val = u2; W(3).val = u3; W(4).val = u4; W(5).val = u5;   W(6).val = u6; W(7).val = u7; W(8).val = u8;
W(9).val = -u8; W(10).val = -u7; W(11).val = -u6; W(12).val = -u5; W(13).val = -u4; W(14).val = -u3; W(15).val = -u2; 
W(16).val = -u1;  
   
   rISol = implicitTransport(rISol, G, dT, rock, fluid, 'wells', W);


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

   OilPr9 = opr(mSol9, -rISol.wellSol(9).flux);              %#ok
   OilPr10 = opr(mSol10, -rISol.wellSol(10).flux);
   OilPr11 = opr(mSol11, -rISol.wellSol(11).flux);            
   OilPr12 = opr(mSol12, -rISol.wellSol(12).flux);
   OilPr13 = opr(mSol13, -rISol.wellSol(13).flux);
   OilPr14 = opr(mSol14, -rISol.wellSol(14).flux);
   OilPr15 = opr(mSol15, -rISol.wellSol(15).flux);
   OilPr16 = opr(mSol16, -rISol.wellSol(16).flux);
   
   po9ocC1 = [po9ocC1; OilPr9];                 %#ok
   po10ocC1 = [po10ocC1; OilPr10];
   po11ocC1 = [po11ocC1; OilPr11];
   po12ocC1 = [po12ocC1; OilPr12];
   po13ocC1 = [po13ocC1; OilPr13];
   po14ocC1 = [po14ocC1; OilPr14];
   po15ocC1 = [po15ocC1; OilPr15];
   po16ocC1 = [po16ocC1; OilPr16];
   
   % Measure flux in production cells at every step (implicit solution)

   WatPr9 = wpr(mSol9, -rISol.wellSol(9).flux);            %#ok
   WatPr10 = wpr(mSol10, -rISol.wellSol(10).flux);
   WatPr11 = wpr(mSol11, -rISol.wellSol(11).flux);
   WatPr12 = wpr(mSol12, -rISol.wellSol(12).flux);
   WatPr13 = wpr(mSol13, -rISol.wellSol(13).flux);
   WatPr14 = wpr(mSol14, -rISol.wellSol(14).flux);
   WatPr15 = wpr(mSol15, -rISol.wellSol(15).flux);
   WatPr16 = wpr(mSol16, -rISol.wellSol(16).flux);
   
   pw9ocC1 = [pw9ocC1; WatPr9];                 %#ok
   pw10ocC1 = [pw10ocC1; WatPr10];
   pw11ocC1 = [pw11ocC1; WatPr11];
   pw12ocC1 = [pw12ocC1; WatPr12];
   pw13ocC1 = [pw13ocC1; WatPr13];
   pw14ocC1 = [pw14ocC1; WatPr14];
   pw15ocC1 = [pw15ocC1; WatPr15];
   pw16ocC1 = [pw16ocC1; WatPr16];
   
   %Measure flux in inyection cell at every step 
   
   WatIn1 = wir(mSol1, rISol.wellSol(1).flux);            %#ok
   WatIn2 = wir(mSol2, rISol.wellSol(2).flux);
   WatIn3 = wir(mSol3, rISol.wellSol(3).flux);
   WatIn4 = wir(mSol4, rISol.wellSol(4).flux);
   WatIn5 = wir(mSol5, rISol.wellSol(5).flux);
   WatIn6 = wir(mSol6, rISol.wellSol(6).flux);
   WatIn7 = wir(mSol7, rISol.wellSol(7).flux);
   WatIn8 = wir(mSol8, rISol.wellSol(8).flux);
   
   uw1 = [uw1; WatIn1];                 
   uw2 = [uw2; WatIn2];
   uw3 = [uw3; WatIn3];
   uw4 = [uw4; WatIn4];
   uw5 = [uw5; WatIn5];
   uw6 = [uw6; WatIn6];
   uw7 = [uw7; WatIn7];
   uw8 = [uw8; WatIn8];
   
   uoc1C1 = [uoc1C1; W(1).val];                 %#ok
   uoc2C1 = [uoc2C1; W(2).val];
   uoc3C1 = [uoc3C1; W(3).val];
   uoc4C1 = [uoc4C1; W(4).val];
   uoc5C1 = [uoc5C1; W(5).val];
   uoc6C1 = [uoc6C1; W(6).val];
   uoc7C1 = [uoc7C1; W(7).val];
   uoc8C1 = [uoc8C1; W(8).val];
   
   %Calculate the NVP value for each step
   
   NPV1oc=NPV1oc+dT*(-(WatIn1+WatIn2+WatIn3+WatIn4+WatIn5+WatIn6+WatIn7+WatIn8)*wi-(WatPr9+WatPr10+WatPr11+WatPr12+ ...
   WatPr13+WatPr14+WatPr15+WatPr16)*wp+(OilPr9+OilPr10+OilPr11+OilPr12+OilPr13+OilPr14+OilPr15+OilPr16)*op)/(1+b)^t;
   NPVocC1=[NPVocC1; NPV1oc];

   % Increase time and continue 
   t = t + dT;
   l=l+1;

end


%%
% Plot the water and oil rates
n = size(po9ocC1,1);

figure
subplot(2,4,1),
   plot(1:n,3600*24*po9ocC1(:,1), 1:n, 3600*24*po9socC1(:,1) )
   title('Oil production in CV P9');
subplot(2,4,2),
   plot(1:n,3600*24*po10ocC1(:,1), 1:n,3600*24*po10socC1(:,1))
   title('Oil production in CV P10');
subplot(2,4,3),
   plot(1:n,3600*24*po11ocC1(:,1), 1:n,3600*24*po11socC1(:,1))
   title('Oil production in CV P11');
subplot(2,4,4),
   plot(1:n,3600*24*po12ocC1(:,1), 1:n,3600*24*po12socC1(:,1))
   title('Oil production in CV P12');
subplot(2,4,5),
   plot(1:n,3600*24*po13ocC1(:,1), 1:n,3600*24*po13socC1(:,1))
   title('Oil production in CV P13');
subplot(2,4,6),
   plot(1:n,3600*24*po14ocC1(:,1), 1:n,3600*24*po14socC1(:,1))
   title('Oil production in CV P14');
subplot(2,4,7),
   plot(1:n,3600*24*po15ocC1(:,1), 1:n,3600*24*po15socC1(:,1))
   title('Oil production in CV P15');
subplot(2,4,8),
   plot(1:n,3600*24*po16ocC1(:,1), 1:n,3600*24*po16socC1(:,1))
   title('Oil production in CV P16');
   
   figure
subplot(2,4,1),
   plot(1:n,3600*24*pw9ocC1(:,1), 1:n,3600*24*pw9socC1(:,1))
   title('Water production in CV P9');
subplot(2,4,2),
   plot(1:n,3600*24*pw10ocC1(:,1), 1:n,3600*24*pw10socC1(:,1))
   title('Water production in CV P10');
subplot(2,4,3),
   plot(1:n,3600*24*pw11ocC1(:,1), 1:n,3600*24*pw11socC1(:,1))
   title('Water production in CV P11');
subplot(2,4,4),
   plot(1:n,3600*24*pw12ocC1(:,1), 1:n,3600*24*pw12socC1(:,1))
   title('Water production in CV P12');
subplot(2,4,5),
   plot(1:n,3600*24*pw13ocC1(:,1), 1:n,3600*24*pw13socC1(:,1))
   title('Water production in CV P13');
subplot(2,4,6),
   plot(1:n,3600*24*pw14ocC1(:,1), 1:n,3600*24*pw14socC1(:,1))
   title('Water production in CV P14');
subplot(2,4,7),
   plot(1:n,3600*24*pw15ocC1(:,1), 1:n,3600*24*pw15socC1(:,1))
   title('Water production in CV P15');
subplot(2,4,8),
   plot(1:n,3600*24*pw16ocC1(:,1), 1:n,3600*24*pw16socC1(:,1))
   title('Water production in CV P16');
   
   
figure
subplot(2,4,1),
   plot(1:n,3600*24*uoc1C1(:,1),1:n, 3600*24*usoc1C1)
   title('Water injection in CV I1');
subplot(2,4,2),
   plot(1:n,3600*24*uoc2C1(:,1),1:n, 3600*24*usoc2C1)
   title('Water injection in CV I2');
subplot(2,4,3),
   plot(1:n,3600*24*uoc3C1(:,1),1:n, 3600*24*usoc3C1)
   title('Water injection in CV I3');
subplot(2,4,4),
   plot(1:n,3600*24*uoc4C1(:,1),1:n, 3600*24*usoc4C1)
   title('Water injection in CV I4');
subplot(2,4,5),
   plot(1:n,3600*24*uoc5C1(:,1),1:n, 3600*24*usoc5C1)
   title('Water injection in CV I5');
subplot(2,4,6),
   plot(1:n,3600*24*uoc6C1(:,1),1:n, 3600*24*usoc6C1)
   title('Water injection in CV I6');
subplot(2,4,7),
   plot(1:n,3600*24*uoc7C1(:,1),1:n, 3600*24*usoc7C1)
   title('Water injection in CV I7');
subplot(2,4,8),
   plot(1:n,3600*24*uoc8C1(:,1),1:n, 3600*24*usoc8C1)
   title('Water injection in CV I8');
   
%Plot the NPV

figure
   plot(1:n,NPVocC1(:,1),1:n, NPVsocC1)
   title('Net Present Value');
   
   po9ocC1 = po9ocC1*24*60*60;
po10ocC1 = po10ocC1*24*60*60;
po11ocC1 = po11ocC1*24*60*60;
po12ocC1 = po12ocC1*24*60*60;
po13ocC1 = po13ocC1*24*60*60;
po14ocC1 = po14ocC1*24*60*60;
po15ocC1 = po15ocC1*24*60*60;
po16ocC1 = po16ocC1*24*60*60;




pw9ocC1 = pw9ocC1*24*60*60;
pw10ocC1 = pw10ocC1*24*60*60;
pw11ocC1 = pw11ocC1*24*60*60;
pw12ocC1 = pw12ocC1*24*60*60;
pw13ocC1 = pw13ocC1*24*60*60;
pw14ocC1 = pw14ocC1*24*60*60;
pw15ocC1 = pw15ocC1*24*60*60;
pw16ocC1 = pw16ocC1*24*60*60;

uoc1C1 = uoc1C1*24*60*60;
uoc2C1 = uoc2C1*24*60*60;
uoc3C1 = uoc3C1*24*60*60;
uoc4C1 = uoc4C1*24*60*60;
uoc5C1 = uoc5C1*24*60*60;
uoc6C1 = uoc6C1*24*60*60;
uoc7C1 = uoc7C1*24*60*60;
uoc8C1 = uoc8C1*24*60*60; 


Tvec = (dT : dT : T)';
oilT1 = cumtrapz ([0; Tvec], [zeros([1, 1]); po9ocC1]);
oilT2 = cumtrapz ([0; Tvec], [zeros([1, 1]); po10ocC1]);
oilT3 = cumtrapz ([0; Tvec], [zeros([1, 1]); po11ocC1]);
oilT4 = cumtrapz ([0; Tvec], [zeros([1, 1]); po12ocC1]);
oilT5 = cumtrapz ([0; Tvec], [zeros([1, 1]); po13ocC1]);
oilT6 = cumtrapz ([0; Tvec], [zeros([1, 1]); po14ocC1]);
oilT7 = cumtrapz ([0; Tvec], [zeros([1, 1]); po15ocC1]);
oilT8 = cumtrapz ([0; Tvec], [zeros([1, 1]); po16ocC1]);
wat1 = cumtrapz ([0; Tvec], [zeros([1, 1]); pw9ocC1]);
wat2 = cumtrapz ([0; Tvec], [zeros([1, 1]); pw10ocC1]);
wat3 = cumtrapz ([0; Tvec], [zeros([1, 1]); pw11ocC1]);
wat4 = cumtrapz ([0; Tvec], [zeros([1, 1]); pw12ocC1]);
wat5 = cumtrapz ([0; Tvec], [zeros([1, 1]); pw13ocC1]);
wat6 = cumtrapz ([0; Tvec], [zeros([1, 1]); pw14ocC1]);
wat7 = cumtrapz ([0; Tvec], [zeros([1, 1]); pw15ocC1]);
wat8 = cumtrapz ([0; Tvec], [zeros([1, 1]); pw16ocC1]);

   totaloil = oilT1+oilT2+oilT3+oilT4+oilT5+oilT6+oilT7+oilT8;
totalwater = wat1+wat2+wat3+wat4+wat5+wat6+wat7+wat8;
%%
displayEndOfDemoMessage(mfilename)
