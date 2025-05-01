% File: generate_pv_dataset.m

% Define the ranges
irr_values = linspace(100, 2000, 20);       % 20 irradiance points
temp_values = linspace(10, 100, 20);         % 20 temperature points

% Initialize dataset
dataset = [];  % [Irradiance, Temperature, 0dc, Duty Cycle, Mod Index]

model_name = 'simulation_on_grid_connected_PV';  % Your .slx model name

% Loop through combinations
for i = 1:length(irr_values)
    for j = 1:length(temp_values)
        ir = irr_values(i);
        temp = temp_values(j);
        
        % Create time vector and inputs for simin blocks
        t = [0; 0.5];  % 0.1s sim time
        irr_input = [t, [ir; ir]];
        temp_input = [t, [temp; temp]];
        
        % Assign to base workspace
        assignin('base', 'simin', irr_input);     % For irradiance simin block
        assignin('base', 'simin1', temp_input);   % For temperature simin block

        % Run the simulation
        simOut = sim(model_name, 'StopTime', '0.5', 'SaveOutput', 'on');

        % Extract outputs (last value)
        try
            %duty = get(simOut, 'duty_output');
            duty = simOut.duty_output.signals.values(end);
            disp(simOut.who);
            %mod_index = get(simOut, 'mod_index_output');
            %vdc = get(simOut, 'vdc_output');
            vdc = simOut.vdc_output.signals.values(end);
            mod_index = simOut.mod_index_output.signals.values(end);


            % Add row to dataset
            dataset = [dataset; ir, temp, vdc, duty, mod_index];
        catch ME
            warning("Error collecting data at Ir=%.1f, Temp=%.1f. Message: %s", ir, temp, ME.message);
        end
    end
end

% Save dataset to xlsx

xlswrite("pvdataset.xlsx", dataset);


disp("✅ Dataset generation complete. File saved as 'pv_dataset.csv'");
