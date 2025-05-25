% MATLAB script to test if phoneme groups are significantly different using variable clusters across frames
clear
clc
close all
if ismac
    homeDir = '/Volumes/lsa-psyc-djbrang/Freesurfer/';
else
    homeDir = '/nfs/turbo/lsa-psyc-djbrang/Freesurfer/';
end
addpath([homeDir 'LipreadingInTheWild/Fathom/'])

% Cody
projDir = [homeDir '/LipreadingInTheWild/TCN_LIPREADING_ROOT/Save_Embeddings/'];
% David
% projDir ='/Users/djbrang/University of Michigan Dropbox/David Brang/MAIN/PROJECTS/DeepLearning_TCN_LIPREADING_ROOT/Save_Embeddings/';

% 500 vis centered on 4th frame, trained to 30 epochs
data_dir= [projDir '/train_logs/tcn/2025-05-01T215804'];

% Define the 18 specific phonemes to keep
selected_phonemes = {'b', 'd', 'dʒ', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 's', 't', 'tʃ', 'w', 'ɹ', 'θ'};

% Define the mapping between phonemes and their example words
selected_words = {
% b
 {'BECAUSE', 'BLACK', 'BUDGET', 'BIGGEST', 'BANKS'},
% d
 {'DEATH', 'DIFFERENCE', 'THERE', 'DAVID', 'DOING'},
% dʒ
 {'GENERAL', 'GEORGE', 'GERMANY', 'JAMES', 'JUDGE'},
% f
 {'FOCUS', 'FIGHT', 'FRANCE', 'FACING', 'FURTHER'},
% g
 {'GREECE', 'GIVEN', 'GUILTY', 'GREAT', 'GIVING'},
% h
 {'HAPPEN', 'HEALTH', 'HOMES', 'HIGHER', 'HUNDREDS'},
% j
 {'EUROPE', 'EUROPEAN', 'UNION', 'UNITED', 'USING'},
% k
 {'COURT', 'KILLED', 'CALLED', 'CAMERON', 'CAMPAIGN'},
% l
 {'LABOUR', 'LOCAL', 'LEVEL', 'LIKELY', 'LARGE'},
% m
 {'MIDDLE', 'MAJOR', 'MEANS', 'MOMENT', 'MEMBER'},
% n
 {'NORTH', 'NUMBER', 'KNOWN', 'NATIONAL', 'NEVER'},
% p
 {'PLACE', 'POINT', 'PARENTS', 'PEOPLE', 'PUBLIC'},
% s
 {'CENTRAL', 'SECOND', 'SIDES', 'SOCIAL', 'SUNDAY'},
% t
 {'THING', 'TODAY', 'TAKEN', 'TEMPERATURES', 'TIMES'},
% tʃ
 {'CHALLENGE', 'CHANCE', 'CHANGE', 'CHANGES', 'CHARGE'},
% w
 {'WEAPONS', 'WAITING', 'WOMEN', 'WHERE', 'WINDS'},
% ɹ
 {'RATES', 'RATHER', 'REALLY', 'REASON', 'RECENT'},
% θ
 {'THINGS', 'THINK', 'THOUGHT', 'THOUSANDS', 'THREAT'}
};

% Create a flat list of all the selected words
all_selected_words = {};
for i = 1:length(selected_words)
    all_selected_words = [all_selected_words; selected_words{i}(:)];
end

% Create mapping between phoneme symbols and their representation in your data
phoneme_mapping = containers.Map();
phoneme_mapping('b') = 'b';
phoneme_mapping('d') = 'd';
phoneme_mapping('dʒ') = 'dzh';
phoneme_mapping('f') = 'f';
phoneme_mapping('g') = 'g';
phoneme_mapping('h') = 'h';
phoneme_mapping('j') = 'j';
phoneme_mapping('k') = 'k';
phoneme_mapping('l') = 'l';
phoneme_mapping('m') = 'm';
phoneme_mapping('n') = 'n';
phoneme_mapping('p') = 'p';
phoneme_mapping('s') = 's';
phoneme_mapping('t') = 't';
phoneme_mapping('tʃ') = 'tsh';
phoneme_mapping('w') = 'w';
phoneme_mapping('ɹ') = 'r';
phoneme_mapping('θ') = 'th';

% Map the IPA phoneme symbols to their representation in your recoded_labels
data_phonemes = cellfun(@(p) phoneme_mapping(p), selected_phonemes, 'UniformOutput', false);

% time_array = arrayfun(@(x) sprintf('%+d', x), -10:10, 'UniformOutput', false);
time_array = arrayfun(@(x) sprintf('%+d', x), -3:0, 'UniformOutput', false);
plot_confmats = 1;

pval_matrix_all = [];
dist_scores_matrix_all = [];

% Step 1: Calculate optimal number of clusters for each frame
% Preallocate arrays to store optimal cluster numbers for each frame
optimal_cluster_nums = zeros(length(time_array), 1);
gap_stats_all = cell(length(time_array), 1);

% First pass: Calculate gap statistics for each frame to determine optimal clusters
disp('Calculating optimal cluster numbers for each frame...');
for timex = 1:length(time_array)
    % Load the CSV for this timepoint
    cd(data_dir)
    timepoint = time_array{timex};
    disp(['Analyzing Frame:' timepoint ' for optimal clusters']);
    
    % Load data 
    warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');
    data = readtable(['stimulus_embeddings__offset',timepoint,'.csv'], 'ReadVariableNames', true, 'VariableNamingRule', 'modify');
    stim_labels = string(data.Stimulus);
    embeddings = table2array(data(:,4:end));
    
    % Filter to selected words
    keep_word_idx = false(size(stim_labels));
    for i = 1:length(all_selected_words)
        keep_word_idx = keep_word_idx | strcmpi(stim_labels, all_selected_words{i});
    end
    filtered_stim_labels = stim_labels(keep_word_idx);
    filtered_embeddings = embeddings(keep_word_idx,:);
    
    % Map words to phonemes
    word_to_phoneme = containers.Map();
    for p = 1:length(selected_phonemes)
        for w = 1:length(selected_words{p})
            word_to_phoneme(upper(selected_words{p}{w})) = data_phonemes{p};
        end
    end
    
    % Create phoneme labels
    phoneme_labels = strings(size(filtered_stim_labels));
    for i = 1:length(filtered_stim_labels)
        word = filtered_stim_labels(i);
        if isKey(word_to_phoneme, word)
            phoneme_labels(i) = word_to_phoneme(word);
        else
            phoneme_labels(i) = "unknown";
        end
    end
    
    % Remove unknowns
    valid_idx = phoneme_labels ~= "unknown";
    labels = phoneme_labels(valid_idx);
    embeddings = filtered_embeddings(valid_idx,:);
    
    % PCA reduction
    embeddings_orig = embeddings;
    [coeff, score, latent] = pca(embeddings);
    k = 20; % Keep top 20 components
    embeddings = score(:,1:k);
    
    % Compute gap statistic to find optimal number of clusters
    % Try different cluster numbers from 2 to 8
    max_clusters = 8;
    min_clusters = 2;
    gaps = zeros(max_clusters-min_clusters+1, 1);
    stds = zeros(max_clusters-min_clusters+1, 1);
    
    % Define options for clustering
    options = statset('Display', 'off', 'MaxIter', 100);
    
    for nc = min_clusters:max_clusters
        gap_obj = evalclusters(embeddings, 'kmeans', 'gap', 'KList', nc, 'B', 50);
        gaps(nc-min_clusters+1) = gap_obj.CriterionValues;
        stds(nc-min_clusters+1) = gap_obj.SE;
    end
    
    % Store gap statistics for this frame
    gap_stats_all{timex} = struct('gaps', gaps, 'stds', stds, 'cluster_range', min_clusters:max_clusters);
    
    % Determine optimal number of clusters using the elbow method
    % Simple approach: find where the rate of improvement slows down
    dgap = diff(gaps);
    d2gap = diff(dgap);
    elbow_idx = find(d2gap > 0, 1, 'first');
    
    if isempty(elbow_idx)
        % If no clear elbow is found, use the maximum gap value
        [~, max_gap_idx] = max(gaps);
        optimal_cluster_nums(timex) = min_clusters + max_gap_idx - 1;
    else
        optimal_cluster_nums(timex) = min_clusters + elbow_idx;
    end
    
    % Provide output about optimal cluster number
    disp(['Optimal number of clusters for Frame ' timepoint ': ' num2str(optimal_cluster_nums(timex))]);
end

% Create a figure to visualize the gap statistics for each frame
figure;
for timex = 1:length(time_array)
    subplot(2, 2, timex);
    
    gap_data = gap_stats_all{timex};
    errorbar(gap_data.cluster_range, gap_data.gaps, gap_data.stds, '-o');
    hold on;
    idx = gap_data.cluster_range == optimal_cluster_nums(timex);
    scatter(optimal_cluster_nums(timex), gap_data.gaps(idx), 100, 'r', 'filled');
    
    title(['Gap Statistic for Frame ' time_array{timex}]);
    xlabel('Number of Clusters');
    ylabel('Gap Statistic');
    grid on;
    xlim([min_clusters-0.5, max_clusters+0.5]);
end
suptitle('Gap Statistics and Optimal Cluster Numbers Across Frames');
saveas(gcf, fullfile(data_dir, 'gap_statistics_across_frames.png'));

% Step 2: Process each frame with its optimal cluster number
for timex = 1:length(time_array)
    % --- Load the CSV ---
    cd(data_dir)
    timepoint = time_array{timex};
    disp(['Processing Frame:' timepoint ' with ' num2str(optimal_cluster_nums(timex)) ' clusters']);
    warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');
    data = readtable(['stimulus_embeddings__offset',timepoint,'.csv'], 'ReadVariableNames', true, 'VariableNamingRule', 'modify');
    stim_labels = string(data.Stimulus);
    embeddings = table2array(data(:,4:end));
    
    % Filter to only include selected words
    keep_word_idx = false(size(stim_labels));
    for i = 1:length(all_selected_words)
        keep_word_idx = keep_word_idx | strcmpi(stim_labels, all_selected_words{i});
    end
    
    % Keep only the selected words
    filtered_stim_labels = stim_labels(keep_word_idx);
    filtered_embeddings = embeddings(keep_word_idx,:);
    
    % Map the filtered words to their corresponding phonemes
    word_to_phoneme = containers.Map();
    for p = 1:length(selected_phonemes)
        for w = 1:length(selected_words{p})
            word_to_phoneme(upper(selected_words{p}{w})) = data_phonemes{p};
        end
    end
    
    % Create phoneme labels for each word
    phoneme_labels = strings(size(filtered_stim_labels));
    for i = 1:length(filtered_stim_labels)
        word = filtered_stim_labels(i);
        if isKey(word_to_phoneme, word)
            phoneme_labels(i) = word_to_phoneme(word);
        else
            phoneme_labels(i) = "unknown";
        end
    end
    
    % Remove any "unknown" labels
    valid_idx = phoneme_labels ~= "unknown";
    labels = phoneme_labels(valid_idx);
    embeddings = filtered_embeddings(valid_idx,:);

    % Step 0: PCA reduce the dimensionality
    embeddings_orig = embeddings;
    [coeff, score, latent] = pca(embeddings);

    % Keep top k components
    k = 20; % or 10
    embeddings = score(:,1:k);

    % Normalize embeddings to use cosine similarity
    embeddings_norm = zeros(size(embeddings));
    for i = 1:size(embeddings, 1)
        embeddings_norm(i,:) = embeddings(i,:) / norm(embeddings(i,:));
    end
    
    % Run MANOVA Globally using normalized embeddings
    [G, groupNames] = findgroups(labels);
    [d, p_global, stats] = manova1(embeddings_norm, labels);
    
    unique_labels = unique(labels);
    nLabels = numel(unique_labels);
    
    % Number of tests to correct
    ntests=((nLabels^2)-nLabels)/2;
    
    pval_matrix = NaN(nLabels);
    dist_scores_matrix= NaN(nLabels);
    cosine_sim_matrix = NaN(nLabels);
    
    for i = 1:nLabels
        for j = i+1:nLabels
            % Correct way to subset with string labels
            idx = ismember(labels, [unique_labels(i), unique_labels(j)]);
            
            % Use normalized embeddings for MANOVA
            emb_sub = embeddings_norm(idx,:);
            labels_sub = labels(idx);
            
            [~, p, stats] = manova1(emb_sub, labels_sub);
            pval_matrix(i,j) = p;
            
            % Wilks' lambda:
            eigenvalues = stats.eigenval;
            dist_scores_matrix(i,j) = prod(1 ./ (1 + eigenvalues));
            
            % Calculate average cosine similarity between groups
            idx_i = find(labels == unique_labels(i));
            idx_j = find(labels == unique_labels(j));
            
            % Get centroids for each group
            centroid_i = mean(embeddings(idx_i,:), 1);
            centroid_j = mean(embeddings(idx_j,:), 1);
            
            % Calculate cosine similarity between centroids
            cosine_sim = dot(centroid_i, centroid_j) / (norm(centroid_i) * norm(centroid_j));
            cosine_sim_matrix(i,j) = cosine_sim;
        end
    end
    
    % Correct for multiple comparisons
    idx = find(~isnan(pval_matrix));
    
    % bonferroni to be cautious
    pval_matrix = pval_matrix.*ntests;
    pval_matrix(pval_matrix>1)=1;
    
    % Fill matrices for symmetry
    for i = 1:nLabels
        for j = i+1:nLabels
            pval_matrix(j,i) = pval_matrix(i,j);
            dist_scores_matrix(j,i) = dist_scores_matrix(i,j);
            cosine_sim_matrix(j,i) = cosine_sim_matrix(i,j);
        end
    end
    
    % Fill in diagonal
    pval_matrix(find(eye(size(pval_matrix))))=1;
    dist_scores_matrix(find(eye(size(dist_scores_matrix))))=1;
    cosine_sim_matrix(find(eye(size(cosine_sim_matrix))))=1;
    
    % Optional: Plot the cosine similarity matrix
    if plot_confmats == 1
        figure;
        imagesc(cosine_sim_matrix);
        colorbar;
        title(['Cosine Similarity Between Phoneme Centroids. Frame:' num2str(timepoint)]);
        xticks(1:nLabels);
        yticks(1:nLabels);
        xticklabels(unique_labels);
        yticklabels(unique_labels);
        xlabel('Label 1');
        ylabel('Label 2');
        axis square;
        caxis([-0.8 1]);
        colormap(parula);
        
        exportgraphics(gcf, ...
            [data_dir '/500Vis_CosineSimilarity_Frame_' timepoint '.png'], ...
            'ContentType', 'image');
        exportgraphics(gcf, ...
            [data_dir '/500Vis_CosineSimilarity_Frame_' timepoint '.eps'], ...
            'ContentType', 'vector');
        close all;
    end

    % Create visualization plots
    if plot_confmats == 1
        figure;
        subplot(2,2,1)
        imagesc(dist_scores_matrix);
        colorbar;
        title('Effect size (Wilks Lambda)');
        xticks(1:nLabels);
        yticks(1:nLabels);
        xticklabels(unique_labels);
        yticklabels(unique_labels);
        xlabel('Label 1');
        ylabel('Label 2');
        axis square;
        caxis([.20 1]);

        subplot(2,2,3)
        imagesc(pval_matrix);
        colorbar;
        title('Pairwise P-value Matrix');
        xticks(1:nLabels);
        yticks(1:nLabels);
        xticklabels(unique_labels);
        yticklabels(unique_labels);
        xlabel('Label 1');
        ylabel('Label 2');
        axis square;
        caxis([0 .05]);
        colormap(flipud(parula));

        % Colormap
        cmap = cbrewer('div','Spectral',length(unique_labels));
        
        % Calculate average embedding for each phoneme label
        average_embedding = nan(length(unique_labels),size(embeddings_orig,2));
        for i=1:length(unique_labels)
            idx = find(unique_labels(i)==labels);
            average_embedding(i,:) = mean(embeddings_orig(idx,:),1);
        end

        % Use variable cluster number for this frame
        num_clusters = optimal_cluster_nums(timex);
        [cluster_idx, cluster_centers] = kmeans(embeddings, num_clusters, 'Replicates', 10);
        
        % Create a mapping from phonemes to clusters
        phoneme_to_cluster = containers.Map();
        for i = 1:length(unique_labels)
            phoneme = unique_labels(i);
            phoneme_indices = find(strcmp(labels, phoneme));
            % Find most common cluster for this phoneme
            phoneme_clusters = cluster_idx(phoneme_indices);
            cluster_counts = histcounts(phoneme_clusters, 1:num_clusters+1);
            [~, most_common_cluster] = max(cluster_counts);
            phoneme_to_cluster(char(phoneme)) = most_common_cluster;
        end
        
        % Display cluster assignments
        disp(['Frame ' timepoint ' phoneme cluster assignments:']);
        cluster_assignments = {};
        for i = 1:length(unique_labels)
            phoneme = char(unique_labels(i));
            cluster = phoneme_to_cluster(phoneme);
            disp([phoneme, ': Cluster ', num2str(cluster)]);
            
            % Store for later analysis
            cluster_assignments{end+1} = struct('frame', timepoint, 'phoneme', phoneme, 'cluster', cluster);
        end
        
        % Add visualization of clusters in the tSNE plot
        subplot(2,2,2)
        % Make a colormap based on clusters
        cluster_cmap = parula(num_clusters);
        marker_colors = cluster_cmap(cluster_idx, :);
        Y = tsne(embeddings_orig);
        scatter(Y(:,1), Y(:,2), 50, marker_colors, 'filled');
        hold on;
        % Add labels
        for i = 1:length(labels)
            text(Y(i,1), Y(i,2), labels(i), 'FontSize', 8);
        end
        title(['TSNE Clusters (', num2str(num_clusters), ' groups)']);

        % t-SNE cluster centers
        subplot(2,2,4)
        Y = tsne(average_embedding);
        gscatter(Y(:,1), Y(:,2), unique_labels, cmap);
        hold on;
        title('TSNE Cluster Centers');
        text(Y(:,1), Y(:,2), unique_labels);
        suptitle(['Frame # ' timepoint ' | ' num2str(num_clusters) ' clusters']);
        
        % Save figure
        exportgraphics(gcf, ...
                [data_dir '/500Vis_4Plot_Frame_' timepoint '_' num2str(num_clusters) 'clusters.png'], ...
                'ContentType', 'image');
        exportgraphics(gcf, ...
                [data_dir '/500Vis_4Plot_Frame_' timepoint '_' num2str(num_clusters) 'clusters.eps'], ...
                'ContentType', 'vector');
        close all;
        
        % Save detailed cluster assignment data
        save([data_dir '/cluster_assignments_frame_' timepoint '.mat'], 'cluster_assignments');
    end

    pval_matrix_all(:,:,timex) = pval_matrix;
    dist_scores_matrix_all(:,:,timex) = dist_scores_matrix;
end

% Create a summary figure showing how cluster numbers evolve across frames
figure;
plot(1:length(time_array), optimal_cluster_nums, '-o', 'LineWidth', 2, 'MarkerSize', 10);
grid on;
title('Optimal Number of Clusters Across Frames');
xlabel('Frame Relative to Speech Onset');
ylabel('Number of Clusters');
xticks(1:length(time_array));
xticklabels(time_array);
ylim([min(optimal_cluster_nums)-0.5, max(optimal_cluster_nums)+0.5]);
saveas(gcf, fullfile(data_dir, 'cluster_evolution_across_frames.png'));

% Save overall results
save([data_dir '/phoneme_clustering_results.mat'], 'optimal_cluster_nums', 'gap_stats_all', 'pval_matrix_all', 'dist_scores_matrix_all', 'time_array');

% Optional: Create a sankey or alluvial diagram showing how phonemes transition between clusters
% This would require additional code using a sankey diagram package