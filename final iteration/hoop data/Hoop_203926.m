%% IMU motion reconstruction
data = OR20250903203926;  % Use your variable

%% Load data
if istable(data)
    data = table2array(data);
end

N = size(data,1);

fs = 120;          % Sampling frequency in Hz
dt = 1/fs;         
t = (0:N-1)'*dt;   % Time vector [Nx1]

f1 = data(:,13);  % Acc_X
f2 = data(:,14);  % Acc_Y
f3 = data(:,15);  % Acc_Z

omega1 = data(:,16); % Gyr_X
omega2 = data(:,17); % Gyr_Y
omega3 = data(:,18); % Gyr_Z

phi   = data(:,3);  % roll
theta = data(:,4);  % pitch
psi   = data(:,5);  % yaw

N = length(f1);
fs = 120;                   % Sampling frequency (Hz)
t = (0:N-1)'/fs;            % Time vector
dt = 1/fs;

%% 3-2-1 Euler anles (integrating for euler angles)
%y0 = [0; 0; 0]; % Initial [psi; theta; phi]
%opts = odeset('RelTol',1e-6,'AbsTol',1e-6);

%[~, Y] = ode45(@(ti,y) euler_321_ODE(ti,y,omega1,omega2,omega3,t), t, y0, opts);

%psi   = Y(:,1);
%theta = Y(:,2);
%phi   = Y(:,3);

%% rotate acceleration into the global frame 
%F_global = zeros(N,3);
%for k = 1:N
    % 3-2-1 rotation matrices
 %   R1 = [cos(psi(k)) sin(psi(k)) 0;
  %       -sin(psi(k)) cos(psi(k)) 0;
   %       0           0           1];

    %R2 = [cos(theta(k)) 0 -sin(theta(k));
     %     0             1 0;
      %    sin(theta(k)) 0 cos(theta(k))];

    %R3 = [1 0 0;
     %     0 cos(phi(k)) sin(phi(k));
      %    0 -sin(phi(k)) cos(phi(k))];

    %R = R3*R2*R1;
    %F_global(k,:) = (R * [f1(k); f2(k); f3(k)])';
%end
%%
% Preallocate
e1 = zeros(3, N);
e2 = zeros(3, N);
e3 = zeros(3, N);
 
accel_vector = zeros(3, N);
 
for i = 1:N
    % Unit vectors
    e1(:,i) = [ cosd(psi(i))*cosd(theta(i));
                sind(psi(i))*cosd(theta(i));
               -sind(theta(i)) ];
 
    e2(:,i) = [  sind(phi(i))*sind(theta(i))*cosd(psi(i)) - sind(psi(i))*cosd(phi(i));
                 sind(phi(i))*sind(psi(i))*sind(theta(i)) + cosd(phi(i))*cosd(psi(i));
                 sind(phi(i))*cosd(theta(i)) ];
 
    e3(:,i) = [  sind(phi(i))*sind(psi(i)) + sind(theta(i))*cosd(phi(i))*cosd(psi(i));
                -sind(phi(i))*cosd(psi(i)) + sind(psi(i))*sind(theta(i))*cosd(phi(i));
                 cosd(phi(i))*cosd(theta(i)) ];
 
    % Acceleration vector
    accel_vector(:,i) = f1(i)*e1(:,i) + f2(i)*e2(:,i) + f3(i)*e3(:,i);
end
%% subtract the mean to compensate for gravity
x1ddot = accel_vector(1,:) - mean(accel_vector(1,:));
x2ddot = accel_vector(2,:) - mean(accel_vector(2,:));
x3ddot = accel_vector(3,:) - mean(accel_vector(3,:));
x1ddot= x1ddot';
x2ddot= x2ddot';
x3ddot= x3ddot';

%% Integrate for velocity and position
x1dot = cumtrapz(t, x1ddot);
x2dot = cumtrapz(t, x2ddot);
x3dot = cumtrapz(t, x3ddot);
figure
hold on
plot(t,x1dot)
plot(t,x2dot)
plot(t,x3dot)
title ('Integrated speed with no detrending')
x1dotnor = detrend_custom(t, x1dot, 'polynomial', 6);
x2dotnor = detrend_custom(t, x2dot, 'polynomial', 6);
x3dotnor = detrend_custom(t, x3dot, 'polynomial', 6);
figure
hold on 
plot(t,x1dotnor)
plot(t,x2dotnor)
plot(t,x3dotnor)
title ('Integrated speed with detrending')
x1 = cumtrapz(t, x1dotnor);
x2 = cumtrapz(t, x2dotnor);
x3 = cumtrapz(t, x3dotnor);
x1nor = detrend_custom(t, x1, 'polynomial', 6);
x2nor = detrend_custom(t, x2, 'polynomial', 6);
x3nor = detrend_custom(t, x3, 'polynomial', 6);


pos = [x1nor, x2nor, x3nor];
%% %Save results

results_file = 'C:\Users\braid\OneDrive\Desktop\Motion Reconstruction\203926_results\hoop_203926_results.mat';

% Store variables in a struct
results = struct();
results.acc_global_hoop      = accel_vector';        % Nx3 global acceleration
results.acc_global_gravity_comp_hoop = [x1ddot, x2ddot, x3ddot]; % gravity-compensated
results.vel_hoop             = [x1dot, x2dot, x3dot];    % integrated velocity
results.vel_detrended_hoop   = [x1dotnor, x2dotnor, x3dotnor]; % detrended velocity
results.pos_hoop             = pos;                   % integrated position
results.pos_detrended_hoop   = [x1nor, x2nor, x3nor]; % detrended position
results.euler_angles_hoop    = [phi, theta, psi];     % Euler angles (degrees)

% Save to MAT file
save(results_file, '-struct', 'results');

disp(['Results saved to: ' results_file]);


%% ---------------------------
% 1. Define hoop geometry (thin torus)
% ---------------------------

R_hoop = 0.415;       % hoop radius (83 cm diameter / 2)
r_thickness = 0.02;   % hoop tube radius (2 cm thickness)

n_circle = 30;   % points around small tube
n_hoop = 100;    % points around main hoop

% Create hoop vertices (mesh)
theta_mesh = linspace(0, 2*pi, n_hoop);    % main hoop angle
phi_mesh   = linspace(0, 2*pi, n_circle);  % tube angle

[Theta_mesh, Phi_mesh] = meshgrid(theta_mesh, phi_mesh);

Xc = (R_hoop + r_thickness*cos(Phi_mesh)) .* cos(Theta_mesh);
Yc = (R_hoop + r_thickness*cos(Phi_mesh)) .* sin(Theta_mesh);
Zc = r_thickness * sin(Phi_mesh);

% Flatten vertices
vertices = [Xc(:), Yc(:), Zc(:)];

% Create faces
faces = convhull(vertices);

% ---------------------------
% 2. Create figure
% ---------------------------
figure('Color','w'); 
axis equal; grid on; view(3);
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Hula Hoop Animation (3-2-1 Euler)');

margin = 0.5;
xlim([min(pos(:,1))-margin, max(pos(:,1))+margin]);
ylim([min(pos(:,2))-margin, max(pos(:,2))+margin]);
zlim([min(pos(:,3))-margin, max(pos(:,3))+margin]);

% Initialize patch
h_hoop = patch('Vertices', vertices, 'Faces', faces, ...
               'FaceColor', [0.8 0.2 0.2], 'EdgeColor', 'k');

%---------------------------
% 3. Animation loop
% ---------------------------
N_anim = length(pos);

disp('Starting hoop animation...');

for i = 1:N_anim
    % Extract Euler angles (degrees)
    phi_i   = phi(i);      % roll
    theta_i = theta(i);    % pitch
    psi_i   = psi(i);      % yaw

    % Rotation matrices (3-2-1 sequence)
    Rx = [1 0 0; 0 cosd(phi_i) sind(phi_i); 0 -sind(phi_i) cosd(phi_i)];
    Ry = [cosd(theta_i) 0 -sind(theta_i); 0 1 0; sind(theta_i) 0 cosd(theta_i)];
    Rz = [cosd(psi_i) sind(psi_i) 0; -sind(psi_i) cosd(psi_i) 0; 0 0 1];

    R = Rz * Ry * Rx;   % combined rotation

    % Rotate around hoop center of mass
    COM = mean(vertices,1);
    vertices_rot = (R * (vertices - COM)')' + pos(i,:);

    % Update patch
    set(h_hoop, 'Vertices', vertices_rot);

    drawnow;
end

disp('Animation finished.');



%% detrending data function
function data_detrended = detrend_custom(time, data, backend, degree, fit)
 
    % Set default arguments
    if nargin < 3 || isempty(backend)
        backend = 'polynomial';
    end
    if nargin < 4 || isempty(degree)
        degree = 6;
    end
    if nargin < 5 || isempty(fit)
        fit = 'linear';
    end
 
    switch backend
        case 'scipy'
            % Equivalent to scipy.signal.detrend(data, type=fit)
            data_detrended = detrend(data, fit);
        case 'polynomial'
            % Polynomial detrending using polyfit
            p = polyfit(time, data, degree);
            trend = polyval(p, time);
            data_detrended = data - trend;
 
        otherwise
            error('Unknown backend: %s. Use "polynomial" or "scipy".', backend);
    end
end

%% ODE function
function dydt = euler_321_ODE(ti, y, omega1, omega2, omega3, tdata)
    w1 = interp1(tdata, omega1, ti);
    w2 = interp1(tdata, omega2, ti);
    w3 = interp1(tdata, omega3, ti);

    psi   = y(1);
    theta = y(2);
    phi   = y(3);

    % Transformation from body rates to Euler angle derivatives (3-2-1)
    T = [0, sin(phi)/cos(theta), cos(phi)/cos(theta);
         0, cos(phi), -sin(phi);
         1, sin(phi)*tan(theta), cos(phi)*tan(theta)];

    dydt = T * [w1; w2; w3];
end