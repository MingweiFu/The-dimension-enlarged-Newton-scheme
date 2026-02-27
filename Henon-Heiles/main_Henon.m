%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/06  Mingwei Fu                                             %%%
%%% This main algorithm perform the Newton iteration for Henon-Heiles  %%%
%%% system. (2d full dimentional tori)                                 %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Dimension-enlarged Newton iteration for the Henon-Heiles system
clear; 
clc;




% Settings ===============================================================
A       = 2;             % Initial truncation parameter
eps     = 0.1;           % Perturbation parameter
mu      = [1; sqrt(2)];  % Basic frequencies of the Henon-Heiles system
a       = [1; 1];        % Fixed amplitudes
R_max   = 5;             % Maximum number of iterations 
tol     = 1e-12;         % Error tolerance
n       = 2;             % Dimension of the torus
% ========================================================================




% initialization =========================================================
ks       = - A : A;
[K1, K2] = ndgrid(ks, ks);
K1_trans = K1.'; 
K2_trans = K2.';
K_vec    = [K1_trans(:), K2_trans(:)];   % Follows row-major order from smallest to largest

L        = (2*A + 1);
z_ini    = zeros(L, L, n);
for j = 1 : n
    ej    = zeros(1, n); 
    ej(j) = 1;
    z_ini(ej(1) + A + 1, ej(2) + A + 1, j) = a(j);  % Map lattice coordinates to matrix indices
end
% ========================================================================




% Call the Newton solver =================================================
[z_history, fre_history, res_history, r] = Newton_HH_Solver(A, eps, mu, a, R_max, tol, n, z_ini);
% ========================================================================




% Construct the approximate function z^{(r)}(t) for each iteration =======
num_steps = r + 1;                % Number of iteration steps
z_t_steps = cell(num_steps, 1); 
q_t_steps = cell(num_steps, 1); 
p_t_steps = cell(num_steps, 1); 

fre_history = [mu, fre_history];  % Append initial frequency

for rr = 1 : num_steps
    % Extract coefficients and frequencies for the current step
    z_hat_r = z_history{rr, 1};  % Fourier coefficient vector at step r    
    mu_r = fre_history(:, rr);   % Approximate frequencies at step r

    % Define and store anonymous functions: z^(r)(t) = sum_k z_hat(k) * exp(i * <k, mu_r> * t)
    Lr = size(z_hat_r, 1);
    Ar_curr = (Lr - 1) / 2;
    
    ks_curr = -Ar_curr : Ar_curr;
    [K1, K2] = ndgrid(ks_curr, ks_curr); 
    K1_vec = K1(:); 
    K2_vec = K2(:);
    z1_vec = z_hat_r(:, :, 1);  z1_vec = z1_vec(:);
    z2_vec = z_hat_r(:, :, 2);  z2_vec = z2_vec(:);
    
    z_t_steps{rr} = @(t) [ (z1_vec.') * exp(1i * (K1_vec*mu_r(1) + K2_vec*mu_r(2)) * t(:).') ; ...
                             (z2_vec.') * exp(1i * (K1_vec*mu_r(1) + K2_vec*mu_r(2)) * t(:).') ];
    q_t_steps{rr} = @(t) sqrt(2) * real(z_t_steps{rr}(t));
    p_t_steps{rr} = @(t) -sqrt(2) * imag(z_t_steps{rr}(t));
end
% ========================================================================




%% Figure 1: Error curve ||F(y^{r}; mu^{r+1})|| decreasing with iteration step r

figure;
semilogy(res_history(1, :), '-o', 'LineWidth', 1.5);
grid on; 
xlabel('iterating steps $r$','Interpreter','latex'); 
ylabel('residual $\| F(y^{(r)}; \omega^{(r)} \|$', 'Interpreter','latex');




%% Figure 2: Convergence of neighboring point error ||z^{r+1} - z^{r}|| with iteration step r

figure('Color', 'w', 'units', 'normalized', 'position', [0.00 0.00 0.6 0.8]);
semilogy(0 : num_steps-2, res_history(2, 1 : num_steps-1), '-s', 'LineWidth', 5, ...
    'MarkerSize', 8, 'MarkerFaceColor', [0,0,1], 'Color', [0,0,1]);

set(gca, 'XTick', 0 : num_steps-2); 
set(gca,'fontsize',40)
xlabel('Iteration: $r$', 'Interpreter', 'latex', 'FontSize', 50); 
ylabel(' $||\hat{z}^{(r+1)} - \hat{z}^{(r)}||$', 'Interpreter', 'latex', 'FontSize', 50);




%% Figure 3: Phase space trajectories of initial and final approximate solutions

% Set evaluation time range and parameters
T_endtime = 20;                         % Time t from 0 to 20
t_eval = linspace(0, T_endtime, 2000); 
t_marks = [0, 10, 20];                  % Time instants for marking points

% Plot settings: index 1 for final step (red), index 2 for initial step (blue)
steps_to_plot = [num_steps, 1]; 
labels = {'Final', 'Initial'};
plot_colors = [1, 0, 0;   % Red (Final)
               0, 0, 1];  % Blue (Initial)
line_styles = {'-', '-'};




% 1. Pre-calculate trajectory data
Q_plot_data = cell(2, 1);
P_plot_data = cell(2, 1);

for i = 1:2
    rr = steps_to_plot(i);
    q_tmp = zeros(2, length(t_eval));
    p_tmp = zeros(2, length(t_eval));
    for ti = 1:length(t_eval)
        q_tmp(:, ti) = q_t_steps{rr}(t_eval(ti));
        p_tmp(:, ti) = p_t_steps{rr}(t_eval(ti));
    end
    Q_plot_data{i} = q_tmp;
    P_plot_data{i} = p_tmp;
end




% 2. Plot phase plane for component 1: q1 - p1
figure('Color', 'w', 'Name', 'Phase Plane 1 (q1-p1)', 'units', 'normalized', 'position', [0.00 0.00 0.6 0.8]);
hold on;

for i = 1:2
    rr = steps_to_plot(i);

    % Plot trajectory lines
    plot(Q_plot_data{i}(1,:), P_plot_data{i}(1,:), 'Color', plot_colors(i,:), ...
         'LineStyle', line_styles{i}, 'LineWidth', 4, 'DisplayName', labels{i});
    
    % Mark points (t = 0, 10, 20) on the final step (i=1) trajectory
    if i == 1
        for tm = t_marks
            q_m = q_t_steps{rr}(tm);
            p_m = p_t_steps{rr}(tm);
            plot(q_m(1), p_m(1), 'o', 'MarkerFaceColor', plot_colors(i,:), ...
                 'MarkerEdgeColor', 'k', 'MarkerSize', 15, 'HandleVisibility', 'off');            
        end
    end
end


% Decoration and axis settings for component 1: q1 - p1
axis([-2, 2, -2, 2]);
axis square; 
box on;

tick_values = linspace(-2, 2, 5);
set(gca, 'XTick', tick_values, 'YTick', tick_values);
set(gca,'fontsize',30)

legend('Location', 'northeast','Interpreter', 'latex', 'FontSize', 30);

text(0.6, 0.0, '$t = 0$', ...
    'Interpreter', 'latex', 'FontSize', 45, 'FontWeight', 'bold')
text(-1.0, 0.6, '$t = 10$', ...
    'Interpreter', 'latex', 'FontSize', 45, 'FontWeight', 'bold')
text(0.1, -1, '$t = 20$', ...
    'Interpreter', 'latex', 'FontSize', 45, 'FontWeight', 'bold')




% 3. Plot phase plane for component 2: q2 - p2 
figure('Color', 'w', 'Name', 'Phase Plane 2 (q2-p2)', 'units', 'normalized', 'position', [0.00 0.00 0.6 0.8]);
hold on;

for i = 1:2
    rr = steps_to_plot(i);

    % Plot trajectory lines
    plot(Q_plot_data{i}(2,:), P_plot_data{i}(2,:), 'Color', plot_colors(i,:), ...
         'LineStyle', line_styles{i}, 'LineWidth', 4, 'DisplayName', labels{i});
    
    % Mark points (t = 0, 10, 20) on the final step (i=1) trajectory
    if i == 1
        for tm = t_marks
            q_m = q_t_steps{rr}(tm);
            p_m = p_t_steps{rr}(tm);
            plot(q_m(2), p_m(2), 'o', 'MarkerFaceColor', plot_colors(i,:), ...
                 'MarkerEdgeColor', 'k', 'MarkerSize', 15, 'HandleVisibility', 'off');           
        end
    end
end

% Decoration and axis settings for component 2: q2 - p2
axis([-2, 2, -2, 2]);
axis square;
box on;

tick_values = linspace(-2, 2, 5);
set(gca, 'XTick', tick_values, 'YTick', tick_values);
set(gca,'fontsize',30)

legend('Location', 'northeast','Interpreter', 'latex', 'FontSize', 30);

text(1.5, 0.3, '$t = 0$', ...
    'Interpreter', 'latex', 'FontSize', 45, 'FontWeight', 'bold')
text(0, -1.7, '$t = 10$', ...
    'Interpreter', 'latex', 'FontSize', 45, 'FontWeight', 'bold')
text(- 1.3, - 0.2, '$t = 20$', ...
    'Interpreter', 'latex', 'FontSize', 45, 'FontWeight', 'bold')




%% Figure 4: Magnitude trajectory of approximate frequency drift |omega^{(r)}|

figure('Color', 'w', 'units', 'normalized', 'position', [0.00 0.00 0.6 0.8]);
actual_fre      = fre_history;
actual_fre_norm = zeros(1, num_steps);
for j = 1 : num_steps
    actual_fre_norm(j) = norm(actual_fre(:, j), 1);
end

plot(0:num_steps-1, actual_fre_norm, '-s', 'LineWidth', 5, ...
     'MarkerSize', 8, 'MarkerFaceColor', [0,0,1], 'Color', [0,0,1]);

ytickformat('%.3f');
set(gca, 'XTick', 0 : num_steps-1);
set(gca,'fontsize',40)
xlabel('Iteration: $r$', 'Interpreter', 'latex', 'FontSize', 50);
ylabel(' $|\omega^{(r)}|$', 'Interpreter', 'latex', 'FontSize', 50);




%% Figure 5: Difference between consecutive approximate frequencies |omega^{(r+1)} - omega^{(r)}|

figure('Color', 'w', 'units', 'normalized', 'position', [0.00 0.00 0.6 0.8]);
fre_diff      = zeros(2, num_steps-1);
fre_diff_norm = zeros(1, num_steps-1);

for r_idx = 1 : num_steps-1
    fre_diff(:, r_idx) = actual_fre(:, r_idx + 1) - actual_fre(:, r_idx);
    fre_diff_norm(r_idx) = norm(fre_diff(:, r_idx), 1);
end

semilogy(0 : num_steps-2, fre_diff_norm, '-s', 'LineWidth', 5, ...
     'MarkerSize', 8, 'MarkerFaceColor', [0,0,1], 'Color', [0,0,1]);

set(gca, 'XTick', 0 : num_steps-2);
set(gca,'fontsize',40)
xlabel('Iteration: $r$', 'Interpreter', 'latex', 'FontSize', 50);
ylabel(' $|\omega^{(r+1)} - \omega^{(r)}|$', 'Interpreter', 'latex', 'FontSize', 50);




%% Figure 6: Magnitude of approximate solution |z^{(r)}(t=10)| at fixed time t = 10

% Set evaluation parameters
t_fixed = 10;  % Fixed time point for evaluation
metric_vals = zeros(1, num_steps);

% Calculate matrix values at t=10 for each step 
for r_idx = 1:num_steps
    z_vec = z_t_steps{r_idx}(t_fixed);
    metric_vals(r_idx) = norm(z_vec, 2); 
end

% Plot curve
figure('Color', 'w', 'Units', 'normalized', 'Position', [0.00 0.00 0.6 0.8]);
plot(0:num_steps-1, metric_vals, '-s', 'LineWidth', 2, ...
     'MarkerSize', 10, 'MarkerFaceColor', [0,0,1], 'Color', [0,0,1]);

% Decoration
grid on;
set(gca, 'XTick', 1:num_steps); 
xlabel('iteration steps $r$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$\sqrt{|z_1(t)|^2 + |z_2(t)|^2}$ at $t=5$', 'Interpreter', 'latex', 'FontSize', 12);




%% Figure 7: Difference of approximate solutions |z^{(r+1)}(10) - z^{(r)}(10)| at fixed time t = 10

% Set evaluation parameters
t_fixed = 10; 
num_diffs = num_steps - 1;
diff_norms = zeros(num_diffs, 1);

% Calculate the L2 norm of the difference between consecutive steps
for r_idx = 1 : num_diffs
    z_curr = z_t_steps{r_idx}(t_fixed);     
    z_next = z_t_steps{r_idx + 1}(t_fixed); 
    
    diff_norms(r_idx) = norm(z_next - z_curr, 2);
end

% Plot curve
figure('Color', 'w', 'Units', 'normalized', 'Position', [0.00 0.00 0.6 0.8]);
semilogy(0:num_diffs-1, diff_norms, '-s', 'LineWidth', 5, ...
         'MarkerSize', 10, 'MarkerFaceColor', [0,0,1], 'Color', [0,0,1]);

% Decoration
set(gca, 'XTick', 0:num_diffs-1); 
set(gca,'fontsize',40)
xlabel('Iteration: $r$', 'Interpreter', 'latex', 'FontSize', 50);
ylabel('$|z^{(r+1)}(10) - z^{(r)}(10)|$', 'Interpreter', 'latex', 'FontSize', 50);



