%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2026/02/05  Mingwei Fu                                             %%%
%%% This main algorithm perform the Newton scheme for 1-d duffing      %%%
%%% oscillator.                                                        %%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Dimension-enlarged Newton iteration for the 1-D Duffing system
clear; 
clc;




% settings ===============================================================
A       = 3;             % Initial truncation parameter  
eps     = 1;             % Perturbation parameter  
mu      = 1;             % Basic frequency of the Duffing system
a       = 1;             % Fixed amplitude 
R_max   = 5;             % Maximum number of iterations 
tol     = 1e-20;         % Error tolerance 
% ========================================================================




% initialization =========================================================
L = 2*A + 1;
idx_1 = 1 + A + 1;  % Vector index idx_1 corresponding to k = 1 in [-A, A]

z_ini = zeros(L, 1);
z_ini(idx_1, 1) = a;  % z(k=1) = a
% ========================================================================




% Call the Newton solver =================================================
[z_history, fre_history, res_history, r] = Newton_duffing_Solver(A, eps, mu, a, R_max, tol, z_ini);
% ========================================================================




% Construct the approximate function z^{(r)}(t) for each iteration =======
num_steps = r + 1;                % Number of iteration steps
z_t_steps = cell(num_steps, 1);   
q_t_steps = cell(num_steps, 1);
p_t_steps = cell(num_steps, 1);

fre_history = [mu, fre_history];  % Append initial frequency 

for rr = 1 : num_steps
    % Extract coefficients and frequency for the current step
    z_hat_r = z_history{rr, 1};      % Fourier coefficient vector at step r
    mu_r    = fre_history(rr);       % Approximate frequency at step r
    
    % Define and store anonymous functions: z^(r)(t) = sum_k z_hat(k) * exp(i * k * mu_r * t)
    Ar = (length(z_hat_r) - 1) / 2;
    k_vec = (-Ar : Ar)'; 
    
    z_t_steps{rr} = @(t) (z_hat_r.') * exp(1i * mu_r * k_vec * t(:).');
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

figure('Color', 'w','units','normalized','position',[0.00 0.00 0.6 0.8]);
semilogy(0 : num_steps-2, res_history(2, 2 : num_steps), '-s', 'LineWidth', 5, ...
    'MarkerSize', 8, 'MarkerFaceColor', [0, 0, 1], 'Color', [0, 0, 1]);

set(gca, 'XTick', 0 : num_steps-2); 
set(gca,'fontsize',40)
xlabel('Iteration: $r$','Interpreter','latex','FontSize', 50); 
ylabel(' $||\hat{z}^{(r+1)} - \hat{z}^{(r)}||$', 'Interpreter','latex','FontSize', 50);
ylim([1e-15, 1e-0]);




%% Figure 3.1: Phase space trajectories of approximate solutions and Hamiltonian contours

% Set evaluation time range
T_endtime = 20;  % Time t from 0 to 20
t_eval = linspace(0, T_endtime, 1000); 

figure('Color', 'w', 'units', 'normalized', 'position', [0.00 0.00 0.6 0.8]);
hold on;

% 1. Plot Hamiltonian (H) contours (black solid line)
q0_appro = sqrt(2) * sum(z_history{num_steps});
p0_appro = 0;
E_total_level = (q0_appro^2 + p0_appro^2) * mu / 2 + eps * q0_appro^4 / 4;
[q_grid, p_grid] = meshgrid(linspace(-2, 2, 300), linspace(-2, 2, 300));
H_total_grid = (q_grid.^2 + p_grid.^2) * mu / 2 + eps * q_grid.^4 / 4;

[~, h_total] = contour(q_grid, p_grid, H_total_grid, [E_total_level E_total_level], ...
    'k-', 'LineWidth', 4, 'DisplayName', '$H$');

% 2. Plot approximate solution trajectories for each iteration 
colors = jet(num_steps); 
for rr = 1:num_steps
    r_val = rr - 1;  % r_val starts from 0
    Q = q_t_steps{rr}(t_eval);
    P = p_t_steps{rr}(t_eval);
    
    % Plot trajectories: dashed lines
    plot(Q, P, '--', 'Color', colors(rr,:), 'LineWidth', 4, ...
         'DisplayName', ['$r = ', num2str(r_val), '$']);
    
    % Mark position at time t = T_endtime / 2 = 10
    Q_mark = q_t_steps{rr}(T_endtime/2);
    P_mark = p_t_steps{rr}(T_endtime/2);
    plot(Q_mark, P_mark, 'o', 'MarkerFaceColor', colors(rr,:), ...
         'MarkerEdgeColor', 'k', 'MarkerSize', 15, 'HandleVisibility', 'off');
end

% 3. Decoration and axis settings
axis([-2, 2, -2, 2]);
axis equal; 
box on;

tick_values = linspace(-2, 2, 5);
set(gca, 'XTick', tick_values, 'YTick', tick_values);

legend('Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 30);

% 4. Plot local zoom-in box and annotations for r = 3 to 5 
target_steps = [4, 5, 6];  % Corresponding to r = 3, 4, 5
q_pts = []; p_pts = [];
for i = 1:length(target_steps)
    curr_step = target_steps(i);
    if curr_step <= num_steps
        q_pts(end+1) = q_t_steps{curr_step}(T_endtime/2);
        p_pts(end+1) = p_t_steps{curr_step}(T_endtime/2);
    end
end

rectangle('Position', [-0.21, -1.60, 0.1, 0.1], ...
          'EdgeColor', 'k', 'LineWidth', 2.2, 'HandleVisibility', 'off');

set(gca,'fontsize',30)

% Add text annotations for r = 3 to 5 in the blank space
text(-1.8, 0.8, '$r = 0$', ...
    'Interpreter', 'latex', 'FontSize', 45, 'FontWeight', 'bold')
text(0.1, 1.7, '$r = 1$', ...
    'Interpreter', 'latex', 'FontSize', 45, 'FontWeight', 'bold')
text(- 1.2, - 1.5, '$r = 2$', ...
    'Interpreter', 'latex', 'FontSize', 45, 'FontWeight', 'bold')
text(-0.5 , -1.75, '$r = 3 \sim 5$', ...
    'Interpreter', 'latex', 'FontSize', 45, 'FontWeight', 'bold')




%% Figure 3.2: Zoomed-in Phase Space (black box region)

figure('Color', 'w', 'units', 'normalized', 'position', [0.00 0.00 0.6 0.8]);
hold on;

axis([-0.26, -0.06, -1.6, -1.4])
axis equal;  
box on;

% Loop to plot trajectories and points at t = T_endtime / 2 = 10
for rr = 4:num_steps
    r_val = rr - 1;
    Q = q_t_steps{rr}(t_eval);
    P = p_t_steps{rr}(t_eval);
    
    % Plot trajectories
    plot(Q, P, '--', 'Color', colors(rr,:), 'LineWidth', 5, ...
         'DisplayName', ['$r = ', num2str(r_val), '$']);
    
    % Mark position at t = T_endtime / 2 = 10
    Q_mark = q_t_steps{rr}(T_endtime/2);
    P_mark = p_t_steps{rr}(T_endtime/2);
    plot(Q_mark, P_mark, 'o', 'MarkerFaceColor', colors(rr,:), ...
         'MarkerEdgeColor', 'k', 'MarkerSize', 15, 'HandleVisibility', 'off');
end

% Decoration and Legend
xlim([-0.21, -0.11]);
ylim([-1.60, -1.50]);
xticks(linspace(-0.21, -0.11, 5));
yticks(linspace(-1.60, -1.50, 5));
ytickformat('%.2f');

legend('Location', 'NorthEast', 'Interpreter', 'latex', 'FontSize', 30);

set(gca,'fontsize',30)

% Add text annotations for r = 3 and r = 4~5 in the blank space
text(- 0.178, - 1.543, '$r = 3$', ...
    'Interpreter', 'latex', 'FontSize', 45, 'FontWeight', 'bold')
text(-0.165, -1.555, '$r = 4 \sim 5$', ...
    'Interpreter', 'latex', 'FontSize', 45, 'FontWeight', 'bold')




%% Figure 4: Trajectory of approximate frequency drift omega^{(r)}

figure('Color', 'w','units','normalized','position',[0.00 0.00 0.6 0.8]);
actual_fre = fre_history(1:num_steps+1);
plot(0:num_steps, actual_fre, '-o', 'LineWidth', 2, ...
     'MarkerSize', 8, 'MarkerFaceColor', '#D95319', 'Color', '#D95319');

grid on;
xlabel('iterating steps $r$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel(' $\omega^{(r)}$', 'Interpreter', 'latex', 'FontSize', 14);




%% Figure 5: Difference between consecutive approximate frequencies |omega^{(r+1)} - omega^{(r)}|

figure('Color', 'w','units','normalized','position',[0.00 0.00 0.6 0.8]);
fre_diff = zeros(num_steps-1, 1);
for r_idx = 1 : num_steps-1
    fre_diff(r_idx) = actual_fre(r_idx + 1) - actual_fre(r_idx);
end

semilogy(0 : num_steps-2, abs(fre_diff), '-s', 'LineWidth', 5, ...
     'MarkerSize', 8, 'MarkerFaceColor', [0, 0, 1], 'Color', [0, 0, 1]);

set(gca, 'XTick', 0 : num_steps-1); 
set(gca,'fontsize',40)
xlabel('Iteration: $r$', 'Interpreter', 'latex', 'FontSize', 50);
ylabel(' $|\omega^{(r+1)} - \omega^{(r)}|$', 'Interpreter', 'latex', 'FontSize', 50);
ylim([1e-8, 1e-0]);
yticks([1e-8, 1e-6, 1e-4, 1e-2, 1e-0]);




%% Figure 6: Convergence of approximate solution magnitude |z^{(r)}(t=10)|

% Set evaluation time point
t_fixed = 10;  % Fixed time point for evaluation
z_vals = zeros(num_steps, 1);

% Calculate magnitude at t=10 for each iteration step
for r_idx = 1:num_steps
    z_vals(r_idx) = z_t_steps{r_idx}(t_fixed);
end

% Plot curve
figure('Color', 'w', 'units', 'normalized', 'position', [0.00 0.00 0.6 0.8]);
plot(0 : num_steps-1, abs(z_vals), '-s', 'LineWidth', 2, ...
     'MarkerSize', 8, 'MarkerFaceColor', [0.2 0.6 0.8], 'Color', [0.2 0.6 0.8]);

% Decoration
grid on;
set(gca, 'XTick', 0 : num_steps-1);
xlabel('Iteration steps $r$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$|z^{(r)}(t=5)|$', 'Interpreter', 'latex', 'FontSize', 12);




%% Figure 7: Difference of approximate solutions at t=10 between consecutive iterations |z^{(r+1)}(t=10) - z^{(r)}(t=10)|

% Calculate the difference at t=10 for each iteration step
z_diff_vals = zeros(num_steps-1, 1);
for r_idx = 1 : num_steps-1
    z_diff_vals(r_idx) = z_vals(r_idx + 1) - z_vals(r_idx);
end

% Plot curve
figure('Color', 'w','units', 'normalized', 'position', [0.00 0.00 0.6 0.8]);
semilogy(0 : num_steps-2, abs(z_diff_vals), '-s', 'LineWidth', 5, ...
     'MarkerSize', 8, 'MarkerFaceColor', [0 0 1], 'Color', [0 0 1]);

% Decoration
set(gca, 'XTick', 0 : num_steps-2); 
set(gca,'fontsize',40)
xlabel('Iteration: $r$', 'Interpreter', 'latex', 'FontSize', 50);
ylabel('$|z^{(r+1)}(10) - z^{(r)}(10)|$', 'Interpreter', 'latex', 'FontSize', 50);
ylim([1e-7, 1e+1]);
yticks([1e-7, 1e-5, 1e-3, 1e-1, 1e+1]);



