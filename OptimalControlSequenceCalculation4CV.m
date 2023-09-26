

% whether or not to show output
verbose = false;
verboseLevel = 2;

%% Define model ------------------------------------------------------------
nx = 20; ny = 20; nz = 5;
G = cartGrid([nx, ny, nz]);
G = computeGeometry(G);
rock.perm  = repmat(100*milli*darcy, [G.cells.num, 1]);
rock.poro  = repmat(0.3, [G.cells.num, 1]);

fluid = initSimpleFluid('mu' , [   1,  10] .* centi*poise     , ...
                        'rho', [1000, 700] .* kilogram/meter^3, ...
                        'n'  , [   2,   2]);
                    
fluid  = adjointFluidFields(fluid);                    

%% Wells and initial rates -------------------------------------------------
totTime = 730*day;

% initial points using material balance

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

%% 8 injection ICVs
%nInj = 8;
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

%

%% System components -------------------------------------------------------
S = computeMimeticIP(G, rock, 'Type', 'comp_hybrid', 'Verbose', verbose, ...
                     'InnerProduct', 'ip_tpf');
W = assembleWellSystem(G, W, 'Type', 'comp_hybrid');

% Initialize --------------------------------------------------------------
state = initResSol(G, 0.0);
state.wellSol = initWellSol(W, 0);

% Objective function ------------------------------------------------------
objectiveFunction = str2func('simpleNPV');

% Initialize schedule and controls ----------------------------------------
numSteps = 300;
schedule = initSchedule(W, 'NumSteps', numSteps, 'TotalTime', ...
                        totTime, 'Verbose', verbose);

% box constraints for each well [min rate, max rate]       
box = [repmat([0 inf], nInj, 1); repmat([-inf 0], nProd, 1)];
controls = initControls(schedule, 'ControllableWells', (1:numel(W)), ...
                                  'MinMax', box, ...
                                  'Verbose', verbose, ...
                                  'NumControlSteps', numSteps, ...
                                  'LinEqConst', {ones(1, numel(W)), 0}');                               

% Run optimization --------------------------------------------------------                              
[simRes, schedule, controls, out] = optimizeObjective(G, S, W, rock, ...
                                        fluid, state, schedule, ...
                                        controls, objectiveFunction, ...
                                        'gradTol',       1e-3, ...
                                        'objChangeTol',  5e-4, ...
                                        'VerboseLevel', verboseLevel);
                                    
%}