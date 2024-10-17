function clusterOptimizationDemo19()
    % Number of nodes
    n_ell = 5;  % Number of nodes
    n_x = 4;    % State dimension

%     % Generalized Gaussian density kernel parameters
%     S = 1.8; 
%     L = 10^(5);
%     eta_SL = 0.5 * S / (L * gamma(1/S));
%     tau = 1 / (L^(S));

    % Generate random data as neighbors' state estimations
    x_hat = randn(n_ell, n_x)*2;

    % Define the adjacency matrix Q (with weights)
    Q = [0   0.2  0  0.1 0.7; 
         0.8  0  0.2  0   0; 
         0   0.5  0  0.5  0; 
         0.7  0  0.1  0  0.2; 
         0.6  0   0  0.4  0];  

    % Convert the adjacency matrix Q to a binary matrix
    Q_binary = Q > 0;

    % Initialize parameters for clustering
    alpha_t = 0.8;  % Increased trust weight for trusted neighbors
    alpha_u = 0.2;  % Decreased trust weight for untrusted neighbors
    beta_t = 0.5;   % Adjusted beta parameters
    beta_u = 0.5;

    % Maximum number of iterations
    maxIter = 100;

    % Create a figure for all plots
    figure;

    % Create subplot for all current nodes
    subplot(2, 3, 1);  % First subplot for all nodes
    hold on;

    % Plot all nodes in black
    scatter(x_hat(:, 1), x_hat(:, 2), 200, 'k', 'filled', 'DisplayName', 'Current Sensor');
    xlabel('$\hat{x}^{1}_{k}$','Interpreter', 'latex','FontSize',18);
    ylabel('$\hat{x}^{2}_{k}$','Interpreter', 'latex','FontSize',18);
    title('(a) All Current Sensors');  % Add (a) to the title
    grid on;
    hold off;

    % Initialize limits for axis scaling
    x_min = min(x_hat(:, 1));
    x_max = max(x_hat(:, 1));
    y_min = min(x_hat(:, 2));
    y_max = max(x_hat(:, 2));

    % Define labels for subplots
    subplot_labels = {'(b)', '(c)', '(d)', '(e)', '(f)'};

    for nodeIdx = 1:n_ell
        current_node = x_hat(nodeIdx, :);
        neighbor_indices = find(Q_binary(nodeIdx, :));  % Find connected neighbors using binary matrix
        other_nodes = x_hat(neighbor_indices, :);

        % Initialize cluster centers for trusted and untrusted neighbors
        %theta_t = mean(other_nodes, 1) + randn(1, n_x) * 0.5;  % Initial trusted neighbors cluster center
        %theta_u = mean(other_nodes, 1) - randn(1, n_x) * 0.5;  % Initial untrusted neighbors cluster center

        theta_t = mean(other_nodes, 1) + randn(1, n_x) * 2;
        theta_u = mean(other_nodes, 1) - randn(1, n_x) * 2;


        % Call the cluster optimization function
        [W_t, W_u, theta_t_final, theta_u_final, b_t, b_u] = clusterOptimization(other_nodes, theta_t, theta_u, alpha_t, alpha_u, beta_t, beta_u, maxIter);
        
        % Plot the results for the current node in its respective subplot
        subplot(2, 3, nodeIdx + 1);  % Create subplot for each node
        hold on;

        % Highlight the current node
        scatter(current_node(1), current_node(2), 200, 'k', 'o', 'filled', 'DisplayName', 'Current Sensor');
        
        % Plot trusted neighbors
        scatter(other_nodes(b_t == 1, 1), other_nodes(b_t == 1, 2), 100, 'b', 'd', 'filled', 'DisplayName', 'Trusted neighbor');

        % Plot untrusted neighbors
        scatter(other_nodes(b_u == 1, 1), other_nodes(b_u == 1, 2), 100, 'r', '^', 'filled', 'DisplayName', 'Untrusted neighbor');

        % Plot cluster centers
        scatter(theta_t_final(1), theta_t_final(2), 200, 'k', 'DisplayName', 'Final trusted center', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'none');
        scatter(theta_u_final(1), theta_u_final(2), 200, 'g', 'DisplayName', 'Final untrusted center', 'MarkerEdgeColor', 'g', 'MarkerFaceColor', 'none');
        
        % Adding labels and title with subplot label
        xlabel('$\hat{x}^{1}_{k}$','Interpreter', 'latex','FontSize',18);
        ylabel('$\hat{x}^{2}_{k}$','Interpreter', 'latex','FontSize',18);
        title([subplot_labels{nodeIdx}, ' Sensor ', num2str(nodeIdx)]); % Add label to title
        grid on;
        hold off;

        % Update limits for axis scaling
        x_min = min(x_min, min([current_node(1), theta_t_final(1), theta_u_final(1)]));
        x_max = max(x_max, max([current_node(1), theta_t_final(1), theta_u_final(1)]));
        y_min = min(y_min, min([current_node(2), theta_t_final(2), theta_u_final(2)]));
        y_max = max(y_max, max([current_node(2), theta_t_final(2), theta_u_final(2)]));
    end

    % Set consistent axis limits for all subplots
    for nodeIdx = 1:n_ell + 1
        subplot(2, 3, nodeIdx);
        xlim([x_min - 1, x_max + 1]);  % Add some padding
        ylim([y_min - 1, y_max + 1]);  % Add some padding
    end
    
    % Add legend only for the second subplot (Node 1)
    subplot(2, 3, 2);
    legend('show');  % Show legend only for the second subplot
end

function [W_t, W_u, theta_t, theta_u, b_t, b_u] = clusterOptimization(x_hat, theta_t, theta_u, alpha_t, alpha_u, beta_t, beta_u, maxIter)
    [n_ell, n_x] = size(x_hat);
    b_t = zeros(n_ell, 1);
    b_u = zeros(n_ell, 1);
    W_t = ones(n_ell, 1);  % Initial trust weights
    W_u = ones(n_ell, 1);  % Initial distrust weights
    G_t = zeros(n_ell, 1);
    G_u = zeros(n_ell, 1);
    
    epsilon = 1e-5;  % Small value to prevent weights from going to zero

    for iter = 1:maxIter
        % Calculate the generalized Gaussian density kernel
        S = 1.8; 
        L = 10^(4);
        eta_SL = 0.5 * S / (L * gamma(1/S));
        tau = 1 / (L^(S));
        
        for j = 1:n_ell
            G_t(j) = eta_SL * exp(-tau * sum(abs(x_hat(j, :) - theta_t) .^ S, 2));
            G_u(j) = eta_SL * exp(-tau * sum(abs(x_hat(j, :) - theta_u) .^ S, 2));
        end

        % Update cluster indicators
        b_t = (W_t .* G_t) >= (W_u .* G_u);
        %b_u = (W_t .* G_t) < (W_u .* G_u);
        
        b_u(b_t == 1) = 0;
        b_u(b_t == 0) = 1;

        % Update weights based on the cluster indicators
        bar_alpha_t = alpha_t / ((1 - alpha_t) * beta_t);
        bar_alpha_u = alpha_u / ((1 - alpha_u) * beta_u);

        for j = 1:n_ell
            if b_t(j) == 1
                D_t = sum(b_t .* G_t);
                W_t(j) = max(exp(-bar_alpha_t * D_t), epsilon);  % Protect weight from going to zero
            else
                W_t(j) = 0;  % Reset weight for untrusted
            end

            if b_u(j) == 1
                D_u = sum(b_u .* G_u);
                W_u(j) = max(exp(-bar_alpha_u * D_u), epsilon);  % Protect weight from going to zero
            else
                W_u(j) = 0;  % Reset weight for trusted
            end
        end

        % Normalize weights
        if sum(W_t) > 0
            W_t = W_t / sum(W_t);
        end
        if sum(W_u) > 0
            W_u = W_u / sum(W_u);
        end

        % Update cluster centers
        theta_t = sum(W_t .* x_hat, 1) / sum(W_t);
        theta_u = sum(W_u .* x_hat, 1) / sum(W_u);

        % Output debug information
        disp(['Iteration: ', num2str(iter)]);
        disp(['W_t: ', num2str(W_t')]);
        disp(['W_u: ', num2str(W_u')]);
        disp(['b_t: ', num2str(b_t')]);
        disp(['b_u: ', num2str(b_u')]);
    end
end
