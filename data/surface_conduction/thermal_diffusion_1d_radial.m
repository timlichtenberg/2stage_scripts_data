% Solve 1D radial thermal diffusion equation using implicit method considering time-dependent surface temperature
% written by Gregor J. Golabek (last update: 23/05/2020)

% Clean up before starting script
clear all; close all; clc;

%% Read disk data from input file
name_file     = 'disk_input.xlsx';
time_disk     = xlsread(name_file,'A1:A201');  % Read time [yr]
temp_disk     = xlsread(name_file,'B1:B201');  % Read disk temperature [K]

%% MODEL SETUP AND PARAMETERS

% Job name
name          = 'tim27';

% Model section or full planetsimal?
full_body     =  true;

% Plot results at specified times?
plotting      =  true;

% Model parameters
total_time_yr = 1.0e7;            % Total run time [yr]
dt_day        = 365.25636;        % Length of timestep [d]
output_time   = 1.0e5;            % Create output after this time [yr]
num_nodes     =  8000;            % Number of nodal points [non-dim.]

% Physical parameters
length        = 18000.;           % Length of model domain [m] (only used when full_body==false)
cond          =     3.;           % Thermal conductivity [W/(m K)]
rho           =  3411.6;          % Density of asteroid material [kg/m^3]
c_P           =  1000.;           % Heat capacity [J/(K kg)]
kappa         = cond/(rho*c_P);   % Thermal diffusivity [m^2/s]

% Planetesimal parameters    
radius        =   3.e5;           % Radius [m] (when full_body==true length of model domain is 2.*radius)
T_init        =   150.;           % Initial internal temperature [K]

% Temperatures of interest
T_iso1        =   573.;           % Serpentine decomposition temperature [K] [Ohtsuka et al., 2009]
T_iso2        =  1223.;           % Amphibolite breakdown temperature [K] [Fu and Elkins-Tanton, 2014]

%% END MODEL SETUP AND PARAMETERS

% Create and go to job output directory
mkdir (name)
cd (name)

% Start timer
tic

disp(' ')
disp('*** Modelling 1D thermal evolution of a planetesimal ***')
disp(' ')

% Unit conversions
yr_to_s       = 3600.*24.*365.25636;   % Conversion yr to s
d_to_s        = 3600.*24.;             % Conversion d to s

% Determine number of timesteps
dt            = dt_day*d_to_s;         % Length of each timestep [s]
total_time    = total_time_yr*yr_to_s; % Total run time [s]
num_step      = round(total_time/dt);  % Calculate total number of timesteps [non-dim.]
n             = 1;                     % Set output counter

% For plotting isotherms of interest
if full_body==false
    x_vector  = [0.00  length];
elseif full_body==true    
    x_vector  = [-radius radius];
end

T_iso_line1   = [T_iso1 T_iso1];
T_iso_line2   = [T_iso2 T_iso2];

% Work with disk data
time_disk     = time_disk*yr_to_s;     % Time from disk models [s]
T_max         = round(max(temp_disk)); % Determine maximum surface temperature [K]

% Discretize domain [m] and compute grid resolution
if full_body==false
    dx                      = length/(num_nodes-1);
elseif full_body==true
    dx                      = 2.0*radius/(num_nodes-1);
end    

disp(['Grid resolution dx: ',num2str(dx),' m']);

% Create vector containing all nodal point positions [m]
if full_body==false
    x_vec                   = 0.0:dx:1.0*length';
elseif full_body==true       
    x_vec                   = -1.0*radius':dx:1.0*radius';
    x_vect                  = x_vec(num_nodes/2+1:1:num_nodes);
end    

% Create vectors for storing maximum penetration depth of isotherms of interest [m]
x_max1                      = size(1:1:num_step,1);
x_max2                      = size(1:1:num_step,1);


% Create initial temperature profile [K]
T(1:1:num_nodes,1)          = T_init;

% Create temperature vector for minimum and maximum temperature at each grid point [K]
T_min_grid(1:1:num_nodes,1) = T_init+100.;
T_max_grid(1:1:num_nodes,1) = T_init;


% Define size of radius vector
if full_body==false  
    rad            = size(num_nodes,1);
    rad(1)         = radius;
    rad(num_nodes) = radius-length;
    
elseif full_body==true
    rad            = size(num_nodes,1);
    rad(1)         = -radius;
    rad(num_nodes) = radius;  
end

% Use constants to compute s_factor [non-dim.]
s_factor                    = kappa*dt./dx^2;

% Define matrix
A                           = sparse(num_nodes,num_nodes);

% Introduce matrix entries along diagonal
A(1,1)                      = 1.;
A(num_nodes,num_nodes)      = 1.;

% Introduce remaining matrix entries
for j=2:1:num_nodes-1
    
    % Radial location [m]
    if full_body==false
        rad(j)     = radius-(j-1)*dx;
    elseif full_body==true
        rad(j)     = -radius+(j-1)*dx;
    end    
    
    A(j,j)     = s_factor/rad(j)^2*((rad(j)+dx./2.)^2+(rad(j)-dx./2.)^2)+1.;
    A(j,j-1)   = -s_factor/rad(j)^2*(rad(j)-dx./2.)^2;
    A(j,j+1)   = -s_factor/rad(j)^2*(rad(j)+dx./2.)^2;
    
end


% Set initial time [s]
time     = 0.;

% Define initial temperature as old temperature for first solution
T_old    = T;


% Loop over time
for i=1:1:num_step
        
    % Set surface boundary condition based on disk results
    T_old(1) = interp1(time_disk, temp_disk, time,'linear');
    
    % For full body set boundary condition based on disk results at both sides
    if full_body==true
        T_old(end) = interp1(time_disk, temp_disk, time,'linear');
    end    
        
    % Set new time [s]
    time  = time+dt;
    
    % Solve for temperature at new timestep
    T_new = A\T_old;
    
    % Find minimum and maximum temperatures at every nodal point
    for k=1:1:num_nodes
            
        % Save minimum temperature experienced at certain depth [K]
        if(T_new(k)<T_min_grid(k))
            T_min_grid(k)  = T_new(k);
        end
        
        % Save maximum temperature experienced at certain depth [K]
        if(T_new(k)>T_max_grid(k))
            T_max_grid(k)  = T_new(k);
        end
    
    end
    
    if full_body==false
        
        % Find maximum depth of penetration by serpentine decomposition temperature isotherm [m]
        if(min(size(find(T_new>=T_iso1)))==0)
            x_max1(i) = 0.000;
        end
        if(min(size(find(T_new>=T_iso1)))>0)
            x_max1(i) = x_vec(max(find(T_new>=T_iso1)));
        end
        
        % Find maximum depth of penetration by amphibolite breakdown isotherm [m]
        if(min(size(find(T_new>=T_iso2)))==0)
            x_max2(i) = 0.000;
        end
        if(min(size(find(T_new>=T_iso2)))>0)
            x_max2(i) = x_vec(max(find(T_new>=T_iso2)));
        end
    
    elseif full_body==true
         
        % Find maximum depth of penetration by serpentine decomposition temperature isotherm [m]
        if(min(size(find(T_new>=T_iso1)))==0)
            x_max1(i) = 0.000;
        end
        if(min(size(find(T_new>=T_iso1)))>0)
            x_max1(i) = radius-x_vect(min(find(T_new(num_nodes/2+1:1:num_nodes)>=T_iso1)));
        end
        
        % Find maximum depth of penetration by amphibolite breakdown isotherm [m]
        if(min(size(find(T_new>=T_iso2)))==0)
            x_max2(i) = 0.000;
        end
        if(min(size(find(T_new>=T_iso2)))>0)
            x_max2(i) = radius-x_vect(min(find(T_new(num_nodes/2+1:1:num_nodes)>=T_iso2)));
        end
    end    
    
        
    % Set new temperature as old temperature for next timestep
    T_old = T_new;
    
    
    % Plot results
    if(plotting==true && i==n*round(output_time/total_time_yr*num_step))
        
        figure(1)
        plot(x_vector,T_iso_line1,':','LineWidth',1.25,'Color',[1.00 0.60 0.00]);
        hold on
        plot(x_vector,T_iso_line2,':','LineWidth',1.25,'Color',[0.40 0.40 0.40]);
        hold on
        plot(x_vec,T_new,'r--','LineWidth',2);
        hold off
        if full_body==false
            xlim([0 length]);
            xlabel('Depth {\itD} [m]','FontSize',16);
        elseif full_body==true    
            xlim([-radius radius]);
            xlabel('Radius {\it R } [m]','FontSize',16);
        end    
        ylim([0 T_max]);
        ylabel('Temperature {\itT} [K]','FontSize',16);
        title(['Time = ',num2str(round(time/yr_to_s)),' yrs'],'FontSize',18);
        if full_body==false
            text(length-0.43*length,T_iso1-50.0,'Srp breakdown','FontSize',14,'Color',[1.00 0.60 0.00]);
            text(length-0.43*length,T_iso2+50.0,'Am breakdown','FontSize',14,'Color',[0.40 0.40 0.40]);
        elseif full_body==true
            text(0.,T_iso1-50.0,'Srp breakdown','FontSize',14,'Color',[1.00 0.60 0.00]);
            text(0.,T_iso2+50.0,'Am breakdown','FontSize',14,'Color',[0.40 0.40 0.40]);   
        end    
        
        set(gcf, 'color', 'white');       % Set background color to white
        
        filename = [name,'_',num2str(round(time/yr_to_s))];
        
        print ('-dpng', '-r300',filename);  % Save figure
        close(figure(1))
        
    end
    
    if(i==n*round(output_time/total_time_yr*num_step))
        
        % Create data files featuring: 
        % - Time [yr]
        % - Max. penetration of 573 K isotherm [m]
        % - Max. penetration of 1223 K isotherm [m]
        %
        % - Grid positions [m]
        % - Current temperatures at each grid point [K]
        % - Max temperature so far at each grid point [K]
        
        % Append data to file
        d_data = [(time/yr_to_s); max(x_max1); max(x_max2)];
        fid = fopen([name,'_d_max.txt'],'a');
        fprintf(fid, '%6.2f %6.2f %6.2f\n',d_data);
        fclose(fid);
        
        % Write current temperature data to file
        temp_data = size(3,num_nodes);
        
        for k=1:1:num_nodes
            temp_data(1,k) = x_vec(k);
            temp_data(2,k) = T_new(k); 
            temp_data(3,k) = T_max_grid(k);
        end
        fid = fopen([name,'_T_',num2str(round(time/yr_to_s)),'.txt'],'wt');
        fprintf(fid, '%10.2f %6.2f %6.2f\n',temp_data);
        fclose(fid);
        
        % Reset output counter
        n = n+1;
    end    
    
end

disp(' ')
disp(['Maximum penetration of ',num2str(T_iso1),' K isotherm: ',num2str(max(x_max1)),' m'])
disp(['Maximum penetration of ',num2str(T_iso2),' K isotherm: ',num2str(max(x_max2)),' m'])
disp(' ')

% Plot minimum and maximum temperatures at all depths
if(plotting==true)
    
        figure(2)
        plot(x_vector,T_iso_line1,':','LineWidth',1.25,'Color',[1.00 0.60 0.00]);
        hold on
        plot(x_vector,T_iso_line2,':','LineWidth',1.25,'Color',[0.40 0.40 0.40]);
        hold on
        plot(x_vec,T_max_grid,'r--','LineWidth',2);
        hold off
        if full_body==false
            xlim([0 length]);
            xlabel('Depth {\itD} [m]','FontSize',16);
        elseif full_body==true    
            xlim([-radius radius]);
            xlabel('Radius {\it R } [m]','FontSize',16);
        end
        ylim([0 T_max]);
        
        ylabel('Temperature {\itT} [K]','FontSize',16);
        title(['Time = ',num2str(round(time/yr_to_s)),' yrs'],'FontSize',18);
        if full_body==false
            text(length-0.43*length,T_iso1-50.0,'Srp breakdown','FontSize',14,'Color',[1.00 0.60 0.00]);
            text(length-0.43*length,T_iso2+50.0,'Am breakdown','FontSize',14,'Color',[0.40 0.40 0.40]);
        elseif full_body==true
            text(0.,T_iso1-50.0,'Srp breakdown','FontSize',14,'Color',[1.00 0.60 0.00]);
            text(0.,T_iso2+50.0,'Am breakdown','FontSize',14,'Color',[0.40 0.40 0.40]);   
        end  
    
        set(gcf, 'color', 'white');       % Set background color to white
        
        filename = [name,'_T_max_final'];
        print ('-dpng', '-r300',filename);  % Save figure
        close(figure(2))
    
end

% Go back to start directory
cd ..

% Stop timer
toc