
%% eight injection ICVs
nInj = 2;
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




%}
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
plotGrid(G, 'FaceColor', 'none'); view(3);
[ht, htxt, hs] = plotWell(G, W, 'radius', 0.1, 'height', 2);
set(htxt, 'FontSize', 16);