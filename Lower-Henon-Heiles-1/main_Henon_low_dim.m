%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/09  Mingwei Fu                                             %%%
%%% This main algorithm performs the Newton scheme for lower           %%%
%%% dimensional quasi-periodic solutions of Henon-Heiles system.       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Dimension-enlarged Newton iteration for lower dimensional quasi-periodic solutions of the Henon-Heiles system
clear; 
clc;




% Settings ===============================================================
A         = 2;                   % Initial truncation parameter 
eps       = 0.5;                 % Perturbation parameter
mu        = [1; sqrt(2)];        % Basic frequencies of the Henon-Heiles system
a         = 1;                   % Fixed amplitudes
R_max     = 6;                   % Maximum number of iterations 
tol       = 1e-12;               % Error tolerance
n         = 2;                   % Dimension of system (n=2)
torus_idx = 1;                   % Index of the chosen base torus     
m         = size(torus_idx, 1);  % Torus dimension, k \in Z^{m}
% ========================================================================




% initialization =========================================================
L = 2*A + 1;
idx_1 = 1 + A + 1;  

z_ini = zeros(L, n);  
z_ini(idx_1, torus_idx) = a;
% ========================================================================




% Call the Newton solver =================================================
[z_history, fre_history, res_history, r] = Newton_HH_low_Solver(A, eps, mu, a, R_max, tol, n, torus_idx, z_ini);
% ========================================================================




% Construct the approximate function z^{(r)}(t) for each iteration =======
num_steps = r + 1;                           % Number of iteration steps
z_t_steps = cell(num_steps, 1);
q_t_steps = cell(num_steps, 1); 
p_t_steps = cell(num_steps, 1); 

fre_history = [mu(torus_idx), fre_history];  % Append initial frequency

for rr = 1 : num_steps
    % Extract coefficients and frequency for the current step
    z_hat_r = z_history{rr};  % Fourier coefficient vector at step r
    mu_r = fre_history(rr);   % Approximate frequency at step r

    % Define and store anonymous functions: z^(r)(t) = sum_k z_hat(k) * exp(i * <k, mu_r> * t)
    Lr = size(z_hat_r, 1);
    Ar_curr = (Lr - 1) / 2;
    ks_vec  = (-Ar_curr : Ar_curr)'; 
    z1_hat  = z_hat_r(:, 1); 
    z2_hat  = z_hat_r(:, 2);
    
    z_t_steps{rr} = @(t) [ (z1_hat.') * exp(1i * ks_vec * mu_r * t(:).') ; ...
                             (z2_hat.') * exp(1i * ks_vec * mu_r * t(:).') ];
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

xlim([0, num_steps-2]);
set(gca, 'XTick', 0 : num_steps-2); 
set(gca,'fontsize',30)
xlabel('Iteration: $r$', 'Interpreter', 'latex', 'FontSize', 30); 
ylabel('$||\hat{z}^{(r+1)} - \hat{z}^{(r)}||$', 'Interpreter', 'latex', 'FontSize', 30);



% %% Figure 3.1: Phase space trajectories of each approximate solutions
% 
% % Set evaluation time range and parameters
% T_endtime = 20;                         % Time t from 0 to 20
% t_eval = linspace(0, T_endtime, 1000); 
% colors = jet(num_steps);
% 
% 
% 
% 
% % 1. Pre-calculate trajectory data
% Q_data = cell(num_steps, 1);
% P_data = cell(num_steps, 1);
% 
% for rr = 1:num_steps
%     q_tmp = zeros(2, length(t_eval));
%     p_tmp = zeros(2, length(t_eval));
%     for ti = 1:length(t_eval)
%         q_tmp(:, ti) = q_t_steps{rr}(t_eval(ti));
%         p_tmp(:, ti) = p_t_steps{rr}(t_eval(ti));
%     end
%     Q_data{rr} = q_tmp;
%     P_data{rr} = p_tmp;
% end
% 
% 
% 
% 
% % 2. Plot phase plane for component 1: q1 - p1
% figure('Color', 'w', 'units', 'normalized', 'position', [0.00 0.00 0.6 0.8]);
% hold on;
% 
% for rr = 1 : num_steps
% 
%     % Plot trajectory lines
%     plot(Q_data{rr}(1,:), P_data{rr}(1,:), 'Color', colors(rr,:), 'LineWidth', 0.5 + rr/num_steps, ...
%          'DisplayName', ['Step ', num2str(rr)]);
%     
%     % Mark position at time t = T_endtime / 2 = 10
%     q_m = q_t_steps{rr}(T_endtime / 2);
%     p_m = p_t_steps{rr}(T_endtime / 2);
%     plot(q_m(1), p_m(1), 'o', 'MarkerFaceColor', colors(rr,:), 'MarkerEdgeColor', 'k', ...
%          'MarkerSize', 6, 'HandleVisibility', 'off');
% end
% 
% % Decoration and axis settings for component 1: q1 - p1
% grid on; axis equal; box on;
% xlabel('$q_1$', 'Interpreter', 'latex', 'FontSize', 14);
% ylabel('$p_1$', 'Interpreter', 'latex', 'FontSize', 14);
% title('Hénon-Heiles Phase Plane: $q_1 - p_1$', 'Interpreter', 'latex');
% legend('Location', 'northeastoutside');
% 
% 
% 
% 
% % 3. Plot phase plane for component 2: q2 - p2 
% figure('Color', 'w', 'units','normalized','position',[0.00 0.00 0.6 0.8]);
% hold on;
% 
% for rr = 1 : num_steps
% 
%     % Plot trajectory lines
%     plot(Q_data{rr}(2,:), P_data{rr}(2,:), 'Color', colors(rr,:), 'LineWidth', 0.5 + rr/num_steps, ...
%          'DisplayName', ['Step ', num2str(rr)]);
%     
%     % Mark position at time t = T_endtime / 2 = 10
%     q_m = q_t_steps{rr}(5);
%     p_m = p_t_steps{rr}(5);
%     plot(q_m(2), p_m(2), 'o', 'MarkerFaceColor', colors(rr,:), 'MarkerEdgeColor', 'k', ...
%          'MarkerSize', 6, 'HandleVisibility', 'off');
% end
% 
% % Decoration and axis settings for component 2: q2 - p2
% grid on; axis equal; box on;
% xlabel('$q_2$', 'Interpreter', 'latex', 'FontSize', 14);
% ylabel('$p_2$', 'Interpreter', 'latex', 'FontSize', 14);
% title('Hénon-Heiles Phase Plane: $q_2 - p_2$', 'Interpreter', 'latex');
% legend('Location', 'northeastoutside');





%% Figure 3.2: Phase space trajectories of initial and final approximate solutions

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
figure('Color', 'w', 'Name', 'Phase Plane 1 (q1-p1)', 'units','normalized','position',[0.00 0.00 0.6 0.8]);
hold on;

for i = 1:2
    rr = steps_to_plot(i);

    % Plot trajectory lines
    plot(Q_plot_data{i}(1,:), P_plot_data{i}(1,:), 'Color', plot_colors(i,:), ...
         'LineStyle', line_styles{i}, 'LineWidth', 3, 'DisplayName', labels{i});
    
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

%%%%%%%%%%%%%%%%%% settings of the first basic torus %%%%%%%%%%%%%%%%%%%%%

% Decoration and axis settings for component 1: q1 - p1
axis([-2, 2, -2, 2]);
axis square; 
box on;

xtick_values = linspace(-2, 2, 5);
ytick_values = linspace(-2, 2, 5);
set(gca, 'XTick', xtick_values, 'YTick', ytick_values);
set(gca,'fontsize',30)
legend('Location', 'northeast','Interpreter', 'latex', 'FontSize', 25);

% Mark positions at time t = 0, 10, 20 (1st torus)
text(0.75, 0.0, '$t = 0$', ...
    'Interpreter', 'latex', 'FontSize', 30, 'FontWeight', 'bold')
text(-1.2, 0.5, '$t = 10$', ...
    'Interpreter', 'latex', 'FontSize', 30, 'FontWeight', 'bold')
text(0.2, -0.95, '$t = 20$', ...
    'Interpreter', 'latex', 'FontSize', 30, 'FontWeight', 'bold')




%%%%%%%%%%%%%%%%%% settings of the second basic torus %%%%%%%%%%%%%%%%%%%%

% % Mark the origin, the unperturbed normal frequency space
% plot(0, 0, 'o', 'MarkerFaceColor', plot_colors(2,:), ...
%                  'MarkerEdgeColor', 'k', 'MarkerSize', 15, 'HandleVisibility', 'off');
% 
% % Decoration and axis settings for component 1: q1 - p1
% axis([-2, 2, -2, 2]);
% axis square; 
% box on;
% 
% xtick_values = linspace(-2, 2, 5);
% ytick_values = linspace(-2, 2, 5);
% set(gca, 'XTick', xtick_values, 'YTick', ytick_values);
% set(gca,'fontsize',30)
% legend('Location', 'northeast','Interpreter', 'latex', 'FontSize', 25);
% 
% % Mark positions at time t = 0, 10, 20 (2nd torus)
% text(0.1, 0.1, '$t = 0, 10, 20$', ...
%     'Interpreter', 'latex', 'FontSize', 30, 'FontWeight', 'bold')




% 3. Plot phase plane for component 2: q2 - p2 
figure('Color', 'w', 'Name', 'Phase Plane 2 (q2-p2)',  'units', 'normalized', 'position',[0.00 0.00 0.6 0.8]);
hold on;

for i = 1:2
    rr = steps_to_plot(i);

    % Plot trajectory lines
    plot(Q_plot_data{i}(2,:), P_plot_data{i}(2,:), 'Color', plot_colors(i,:), ...
         'LineStyle', line_styles{i}, 'LineWidth', 3, 'DisplayName', labels{i});
    
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

%%%%%%%%%%%%%%%%%% settings of the first basic torus %%%%%%%%%%%%%%%%%%%%%

% Mark the origin, the unperturbed normal frequency space
plot(0, 0, 'o', 'MarkerFaceColor', plot_colors(2,:), ...
                 'MarkerEdgeColor', 'k', 'MarkerSize', 15, 'HandleVisibility', 'off');

% Decoration and axis settings for component 2: q2 - p2
axis([-2, 2, -2, 2]);
axis square; 
box on;
xtickformat('%.1f');
ytickformat('%.1f');

xtick_values = linspace(-2, 2, 5);
ytick_values = linspace(-2, 2, 5);
set(gca, 'XTick', xtick_values, 'YTick', ytick_values);
set(gca,'fontsize',30)
legend('Location', 'northeast','Interpreter', 'latex', 'FontSize', 25);

% Mark positions at time t = 0, 10, 20 (1st torus)
text(0.3, 0.1, '$t = 0$', ...
    'Interpreter', 'latex', 'FontSize', 30, 'FontWeight', 'bold')
text(0.15, -0.6, '$t = 10$', ...
    'Interpreter', 'latex', 'FontSize', 30, 'FontWeight', 'bold')
text(-1.0, -0.9, '$t = 20$', ...
    'Interpreter', 'latex', 'FontSize', 30, 'FontWeight', 'bold')




%%%%%%%%%%%%%%%%%% settings of the second basic torus %%%%%%%%%%%%%%%%%%%%

% % Decoration and axis settings for component 2: q2 - p2
% axis([-2, 2, -2, 2]);
% axis square; 
% box on;
% 
% xtick_values = linspace(-2, 2, 5);
% ytick_values = linspace(-2, 2, 5);
% set(gca, 'XTick', xtick_values, 'YTick', ytick_values);
% set(gca,'fontsize',30)
% legend('Location', 'northeast','Interpreter', 'latex', 'FontSize', 25);
% 
% % Mark positions at time t = 0, 10, 20 (2nd torus)
% text(1.4, -0.9, '$t = 0$', ...
%     'Interpreter', 'latex', 'FontSize', 30, 'FontWeight', 'bold')
% text(0.6, 0.5, '$t = 10$', ...
%     'Interpreter', 'latex', 'FontSize', 30, 'FontWeight', 'bold')
% text(0.1, 1.0, '$t = 20$', ...
%     'Interpreter', 'latex', 'FontSize', 30, 'FontWeight', 'bold')




%% Figure 4: Magnitude trajectory of approximate frequency drift |omega^{(r)}|

figure('Color', 'w', 'units', 'normalized', 'position',[0.00 0.00 0.6 0.8]);
actual_fre      = fre_history;
actual_fre_norm = zeros(1, num_steps);
for j = 1 : num_steps
    actual_fre_norm(j) = norm(actual_fre(:, j), 1);
end

plot(0:num_steps-1, actual_fre_norm, '-s', 'LineWidth', 5, ...
     'MarkerSize', 8, 'MarkerFaceColor', [0,0,1], 'Color', [0,0,1]);


xlim([0, num_steps-1]);
ytickformat('%.2f');

set(gca, 'XTick', 0 : num_steps-1); 
set(gca,'fontsize',30)
xlabel('Iteration: $r$', 'Interpreter', 'latex', 'FontSize', 30);
ylabel(' $|\omega^{(r)}|$', 'Interpreter', 'latex', 'FontSize', 30);




%% Figure 5: Difference between consecutive approximate frequencies |omega^{(r+1)} - omega^{(r)}|

figure('Color', 'w', 'units', 'normalized', 'position', [0.00 0.00 0.6 0.8]);
fre_diff      = zeros(m, num_steps-1);
fre_diff_norm = zeros(1, num_steps-1);

for r_idx = 1 : num_steps-1
    fre_diff(:, r_idx) = actual_fre(:, r_idx + 1) - actual_fre(:, r_idx);
    fre_diff_norm(r_idx) = norm(fre_diff(:, r_idx), 1);
end

semilogy(0 : num_steps-2, fre_diff_norm, '-s', 'LineWidth', 5, ...
     'MarkerSize', 8, 'MarkerFaceColor', [0,0,1], 'Color', [0,0,1]);

yticks([1e-8, 1e-6, 1e-4, 1e-2, 1e-0]);
set(gca, 'XTick', 0 : num_steps-2); 
set(gca,'fontsize',30)
xlabel('Iteration: $r$', 'Interpreter', 'latex', 'FontSize', 30);
ylabel(' $|\omega^{(r+1)} - \omega^{(r)}|$', 'Interpreter', 'latex', 'FontSize', 30);




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
plot(0 : num_steps-1, metric_vals, '-s', 'LineWidth', 5, ...
     'MarkerSize', 8, 'MarkerFaceColor', [0,0,1], 'Color', [0,0,1]);

% Decoration
xlim([0, num_steps-1]);
set(gca, 'XTick', 0 : num_steps-1); 
set(gca,'fontsize',30)
xlabel('Iteration: $r$', 'Interpreter', 'latex', 'FontSize', 30);
ylabel('$|z^{(r)}(10)| $', 'Interpreter', 'latex', 'FontSize', 30);




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
semilogy( 0 : num_steps - 2, diff_norms, '-s', 'LineWidth', 5, ...
         'MarkerSize', 8, 'MarkerFaceColor', [0,0,1], 'Color', [0,0,1]);

% Decoration
ylim([1e-7, 1e+1]);
yticks([1e-7, 1e-5, 1e-3, 1e-1, 1e+1]);
set(gca, 'XTick', 0 : num_steps-2); 
set(gca,'fontsize',30)
xlabel('Iteration: $r$', 'Interpreter', 'latex', 'FontSize', 30);
ylabel('$|z^{(r+1)}(10) - z^{(r)}(10)|$', 'Interpreter', 'latex', 'FontSize', 30);





