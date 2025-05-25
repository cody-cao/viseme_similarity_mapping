% MATLAB script to test if two (or more) label groups are significantly different
% based on full pairwise distances
clear
clc
close all
% if ismac
%     homeDir = '/Volumes/lsa-psyc-djbrang/Freesurfer/';
% else
%     homeDir = '/nfs/turbo/lsa-psyc-djbrang/Freesurfer/';
% end
pairwiseTest = 'MANOVA'; %'permANOVA'
% addpath([homeDir 'LipreadingInTheWild/Fathom/'])

% MATLAB script to test if two (or more) label groups are significantly different
% based on full pairwise distances
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


% Need to check for repeats!
% test/WOMEN_00011.npz and test/WOMEN_00014.npz are actually the same video
% I watched to confirm and the associated .txt files have the same identifier Disk reference: 6245395697430738233
% we can remove duplicates from the distance matrices since they'll be zero
% At least it's a nice confirmation


% Maybe load all the text and pull Disk reference numbers
% Good to also double check no matching reference numbers in train/val and test


% Also be wary about going too many frames back since videos shifted frame 10
% Long words more likely to have had zero padding - changed in new version to edge
% May want to retrain A and V and move speech onset later (frame 15/29)


% One thing we can try is to counter-act the bias with a baseline
% E.g., by subtracting the pairwise distance from -5 to -10.



% --- Load the CSV ---
% 18 vis
% data_dir = '/Users/djbrang/University of Michigan Dropbox/David Brang/MAIN/PROJECTS/DeepLearning_TCN_LIPREADING_ROOT/Save_Embeddings/train_logs/tcn/2025-04-27T130851';

% 18 vis with soft zeroing 12-29
% data_dir = '/Users/djbrang/University of Michigan Dropbox/David Brang/MAIN/PROJECTS/DeepLearning_TCN_LIPREADING_ROOT/Save_Embeddings/train_logs/tcn/2025-04-28T110801';

% 18 vis with soft zeroing 11-29
% data_dir = '/Users/djbrang/University of Michigan Dropbox/David Brang/MAIN/PROJECTS/DeepLearning_TCN_LIPREADING_ROOT/Save_Embeddings/train_logs/tcn/2025-04-28T123033';

% 18 vis with soft zeroing 6-29 as test
% data_dir = '/Users/djbrang/University of Michigan Dropbox/David Brang/MAIN/PROJECTS/DeepLearning_TCN_LIPREADING_ROOT/Save_Embeddings/train_logs/tcn/2025-04-28T123338';

% 18 vis. reflect padding. fixed np.roll boundary. retrained to epoch 30.
% data_dir ='/Users/djbrang/University of Michigan Dropbox/David Brang/MAIN/PROJECTS/DeepLearning_TCN_LIPREADING_ROOT/Save_Embeddings/train_logs/tcn/2025-04-29T144118';

% % 18 vis. edge padding. fixed np.roll boundary. retrained to epoch 25.
% data_dir ='/Users/djbrang/University of Michigan Dropbox/David Brang/MAIN/PROJECTS/DeepLearning_TCN_LIPREADING_ROOT/Save_Embeddings/train_logs/tcn/2025-04-29T161457';

% 18 vis. center on 4th frame ([3]) edge padding. fixed np.roll boundary. retrained to epoch 20.
% data_dir ='/Users/djbrang/University of Michigan Dropbox/David Brang/MAIN/PROJECTS/DeepLearning_TCN_LIPREADING_ROOT/Save_Embeddings/train_logs/tcn/2025-04-29T174445';

% 18 vis. center on 4th frame ([3]). noise in 5-29.
% data_dir =[projDir '/train_logs/tcn/2025-04-29T182603'];



% % % % % % load in the csv that has phoneme labels. use that instead of labels
% % % % % % For 18 stim, recode
% % % % % recoded_labels = repmat(["B","M","N","P","S","W"],150,1); % 3 words x 50 each
% % % % % recoded_labels = recoded_labels(:);
% % % % % rm_hits = [];


% % 500 vis. soft zeroing 12-29 . speech onset at 11
% data_dir = '/Users/djbrang/University of Michigan Dropbox/David Brang/MAIN/PROJECTS/DeepLearning_TCN_LIPREADING_ROOT/Save_Embeddings/train_logs/tcn/2025-04-28T133008';

% 500 vis. center on 4th frame ([3]). train to 8 epochs
% data_dir = [projDir '/train_logs/tcn/2025-04-30T084159'];

% 500 vis. center on 4th frame ([3]). train to 8 epochs. soft zeroing 6-29 [5:29]
% data_dir = [projDir '/train_logs/tcn/2025-04-30T09:17:59'];

% 500 vis centered on 4th frame, trained to 30 epochs
data_dir= [projDir '/train_logs/tcn/2025-05-01T215804'];


% 500 aud. center on 4th frame ([3]). train to 14 epochs
% data_dir = '/Users/djbrang/University of Michigan Dropbox/David Brang/MAIN/PROJECTS/DeepLearning_TCN_LIPREADING_ROOT/Save_Embeddings/train_logs/tcn/2025-04-30T081927';


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

% 500 from Keyword_Phonemes
recoded_labels = {'a'	'a'	'a'	'a'	'a'	'ah'	'ah'	'ah'	'ah'	'ah'	'ai'	'aw'	'aw'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'd'	'd'	'd'	'd'	'd'	'd'	'd'	'd'	'd'	'd'	'd'	'd'	'd'	'd'	'd'	'dh'	'dh'	'dh'	'dh'	'dh'	'dzh'	'dzh'	'dzh'	'dzh'	'dzh'	'dzh'	'E'	'E'	'E'	'E'	'E'	'E'	'E'	'E'	'E'	'E'	'E'	'E'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'g'	'g'	'g'	'g'	'g'	'g'	'g'	'g'	'g'	'g'	'g'	'g'	'g'	'g'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'i'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'j'	'j'	'j'	'j'	'j'	'j'	'j'	'j'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'kw'	'kw'	'kw'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'long schwa'	'long_i'	'long_i'	'long_i'	'long_i'	'longQ'	'longQ'	'longQ'	'longQ'	'longQ'	'longQ'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'n'	'n'	'n'	'n'	'n'	'n'	'n'	'n'	'n'	'n'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'Q'	'Q'	'Q'	'Q'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'sh'	'sh'	't'	't'	't'	't'	't'	't'	't'	't'	't'	't'	't'	't'	't'	't'	'th'	'th'	'th'	'th'	'th'	'th'	'th'	'th'	'th'	'tsh'	'tsh'	'tsh'	'tsh'	'tsh'	'tsh'	'tsh'	'tsh'	'tsh'	'tsh'	'tsh'	'tsh'	'v'	'v'	'v'	'v'	'v'	'v'	'v'	'v'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'};
recoded_labels = repmat(string(recoded_labels),50,1);
recoded_labels = recoded_labels(:);

% Create an index of samples to keep based on the mapped phoneme representations
keep_idx = false(size(recoded_labels));
for i = 1:length(data_phonemes)
    keep_idx = keep_idx | strcmp(recoded_labels, data_phonemes{i});
end

% Find indices to remove (all that aren't in our 18 selected phonemes)
rm_hits = find(~keep_idx);

% time_array = arrayfun(@(x) sprintf('%+d', x), -10:10, 'UniformOutput', false);
time_array = arrayfun(@(x) sprintf('%+d', x), -3:0, 'UniformOutput', false);
plot_confmats = 1;

pval_matrix_all = [];
dist_scores_matrix_all = [];

for timex = [4 3 2 1] %1:4%1:length(time_array)

    % Optimized MATLAB script to test if label groups are significantly different
    % based on full pairwise distances without nested loops, now for all class pairs
    
    % --- Load the CSV ---
    cd(data_dir)
    timepoint = time_array{timex};
    disp(['Frame:' timepoint])
    warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');
    data = readtable(['stimulus_embeddings__offset',timepoint,'.csv'], 'ReadVariableNames', true, 'VariableNamingRule', 'modify');
    stim_labels = string(data.Stimulus);        % force labels into string type
    embeddings = table2array(data(:,4:end));

    
    % Use grouped labels
%     recoded_labels(rm_hits) = [];
%     embeddings(rm_hits,:)=[];
%     labels=recoded_labels;

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
            % Handle case where word isn't in the mapping (shouldn't happen)
            phoneme_labels(i) = "unknown";
        end
    end
    
    % Remove any "unknown" labels
    valid_idx = phoneme_labels ~= "unknown";
    labels = phoneme_labels(valid_idx);
    embeddings = filtered_embeddings(valid_idx,:);


    % % As a test, only compare BEING, MEASURES, SENIOR, WEATHER
    % labels = labels([1:50, 151:200,601:650,751:800]);
    % embeddings = embeddings([1:50, 151:200,601:650,751:800],:);

    % % 500 word stim. Select a few as tests
    % test_array = ["BEFORE","BEING","FIGHT","FIRST","MEETING","MIGHT","PEOPLE","PERIOD","SEVERAL","SINCE"];
    % idx = ismember(labels, test_array);
    % labels = labels(idx);
    % embeddings = embeddings(idx,:);
    % 

    % Step 0: PCA reduce the dimensionality
    embeddings_orig = embeddings;
%     [coeff, score, latent] = pca(embeddings);
    [coeff, score, latent, ~, explained] = pca(embeddings);
    cumulative_explained = cumsum(explained);
    fprintf('Variance explained by first 20 components: %.2f%%\n', cumulative_explained(20));
    % Keep top k components
    k = 20; % or 10
    embeddings = score(:,1:k);  % Now 100 x k instead of 100 x 512

    % embeddings: 100x512 matrix
    % labels: 100x1 vector (e.g., 0 and 1)
    if strcmp(pairwiseTest,'MANOVA')
%         % Run MANOVA Globally
%         [G, groupNames] = findgroups(labels);
%         [d, p_global, stats] = manova1(embeddings, labels);
%         % disp(['Global MANOVA p-value: ', num2str(p_global)]);
%     
%         unique_labels = unique(labels);
%         nLabels = numel(unique_labels);
%     
%         % Number of tests to correct
%         ntests=((nLabels^2)-nLabels)/2;
%     
%         pval_matrix = NaN(nLabels);
%         dist_scores_matrix= NaN(nLabels);
%     
%         for i = 1:nLabels
%             for j = i+1:nLabels
%                 % Correct way to subset with string labels
%                 idx = ismember(labels, [unique_labels(i), unique_labels(j)]);
%     
%                 emb_sub = embeddings(idx,:);
%                 labels_sub = labels(idx);
%     
%                 [~, p, stats] = manova1(emb_sub, labels_sub);
%                 pval_matrix(i,j) = p;
%     
%                 % Wilks' lambda:
%                 eigenvalues = stats.eigenval;
%                 dist_scores_matrix(i,j) = prod(1 ./ (1 + eigenvalues));
%                
%             end
%         end
% 
%         % Correct
%         idx = find(~isnan(pval_matrix));
%         % [h, crit_p, adj_ci_cvrg, adj_p]=fdr_bh(pval_matrix(idx));
%         % pval_matrix_adj = pval_matrix;
%         % pval_matrix(idx) = adj_p;
%     
%         % bonferroni to be cautious
%         pval_matrix = pval_matrix.*ntests;
%         pval_matrix(pval_matrix>1)=1;
% 
%         % After the loops
%         for i = 1:nLabels
%             for j = i+1:nLabels
%                 pval_matrix(j,i) = pval_matrix(i,j);  % Copy upper triangle to lower triangle
%                 dist_scores_matrix(j,i) = dist_scores_matrix(i,j);  % Copy upper triangle to lower triangle
%             end
%         end
%         % Fill in diagonal
%         pval_matrix(find(eye(size(pval_matrix))))=1;
%         dist_scores_matrix(find(eye(size(dist_scores_matrix))))=1;
        % Normalize embeddings to use cosine similarity
        embeddings_norm = zeros(size(embeddings));
        for i = 1:size(embeddings, 1)
            embeddings_norm(i,:) = embeddings(i,:) / norm(embeddings(i,:));
        end
        
        % Run MANOVA Globally using normalized embeddings
        [G, groupNames] = findgroups(labels);
        [d, p_global, stats] = manova1(embeddings_norm, labels);
        % disp(['Global MANOVA p-value: ', num2str(p_global)]);
        
        unique_labels = unique(labels);
        nLabels = numel(unique_labels);
        
        % Number of tests to correct
        ntests=((nLabels^2)-nLabels)/2;
        
        pval_matrix = NaN(nLabels);
        dist_scores_matrix= NaN(nLabels);
        cosine_sim_matrix = NaN(nLabels); % New matrix to store average cosine similarities
        
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
        
        % Correct
        idx = find(~isnan(pval_matrix));
        % [h, crit_p, adj_ci_cvrg, adj_p]=fdr_bh(pval_matrix(idx));
        % pval_matrix_adj = pval_matrix;
        % pval_matrix(idx) = adj_p;
        
        % bonferroni to be cautious
        pval_matrix = pval_matrix.*ntests;
        pval_matrix(pval_matrix>1)=1;
        
        % After the loops
        for i = 1:nLabels
            for j = i+1:nLabels
                pval_matrix(j,i) = pval_matrix(i,j); % Copy upper triangle to lower triangle
                dist_scores_matrix(j,i) = dist_scores_matrix(i,j); % Copy upper triangle to lower triangle
                cosine_sim_matrix(j,i) = cosine_sim_matrix(i,j); % Copy cosine similarity values
            end
        end
        
        % Fill in diagonal
        pval_matrix(find(eye(size(pval_matrix))))=1;
        dist_scores_matrix(find(eye(size(dist_scores_matrix))))=1;
        cosine_sim_matrix(find(eye(size(cosine_sim_matrix))))=1; % Self-similarity is 1
        
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
            caxis([-0.8 1])
            colormap(parula); % Use a colormap where higher values (more similar) are warmer colors
            
            % exportgraphics(gcf, ...
            %     ['/Volumes/lsa-psyc-djbrang/Freesurfer/LipreadingInTheWild/TCN_LIPREADING_ROOT/Save_Embeddings/500Vis_CosineSimilarity_Frame_' timepoint '.png'], ...
            %     'ContentType', 'image');
            % exportgraphics(gcf, ...
            %     ['/Volumes/lsa-psyc-djbrang/Freesurfer/LipreadingInTheWild/TCN_LIPREADING_ROOT/Save_Embeddings/500Vis_CosineSimilarity_Frame_' timepoint '.eps'], ...
            %     'ContentType', 'vector');
            % close all
        end
%  %% permANOVA below
%  
%     else
%         % Initialize matrices for results
%         unique_labels = unique(labels);
%         nLabels = numel(unique_labels);
%         pval_matrix = NaN(nLabels);
%         F_stat_matrix = NaN(nLabels);
%         dist_scores_matrix = NaN(nLabels); % Add this line to initialize dist_scores_matrix
%         num_permutations = 1000;
%         
%         % Calculate total number of tests for progress bar
%         total_tests = (nLabels * (nLabels - 1)) / 2;
%         test_counter = 0;
%         
%         % Create a progress bar
%         h = waitbar(0, 'Starting PERMANOVA analysis...', 'Name', 'PERMANOVA Progress');
%         
%         % Calculate covariance matrix for Mahalanobis distance
%         covMat = cov(embeddings);
%         
%         % Ensure covariance matrix is well-conditioned
%         if rcond(covMat) < 1e-10
%             % Add small constant to diagonal for regularization
%             covMat = covMat + eye(size(covMat)) * 1e-6;
%         end
%         
%         % Calculate Mahalanobis distance matrix
%         distMatrix = pdist2(embeddings, embeddings, 'mahalanobis', covMat);
%         
%         try
%             for i = 1:nLabels
%                 for j = i+1:nLabels
%                     % Update progress bar
%                     test_counter = test_counter + 1;
%                     progress = test_counter / total_tests;
%                     waitbar(progress, h, sprintf('Running test %d of %d (%.1f%%)', test_counter, total_tests, progress*100));
%                     
%                     % Subset data for just these two groups
%                     idx = ismember(labels, [unique_labels(i), unique_labels(j)]);
%                     
%                     % Extract the distance sub-matrix for just these samples
%                     subDist = distMatrix(idx, idx);
%                     
%                     % Create group labels (1's and 2's)
%                     groupLabels = ones(sum(idx), 1);
%                     groupLabels(labels(idx) == unique_labels(j)) = 2;
%                     
%                     % Run PERMANOVA test
%                     result = f_permanova(subDist, groupLabels, num_permutations, 0); % 0 = no verbose output
%                     
%                     % Store results
%                     pval_matrix(i,j) = result.p;
%                     F_stat_matrix(i,j) = result.F;
%                 end
%             end
%             
%             % Close the progress bar
%             close(h);
%             
%             % Fill in the lower triangle of the matrices
%             for i = 1:nLabels
%                 for j = i+1:nLabels
%                     pval_matrix(j,i) = pval_matrix(i,j);
%                     F_stat_matrix(j,i) = F_stat_matrix(i,j);
%                 end
%             end
%             
%             % Fill diagonal
%             pval_matrix(find(eye(size(pval_matrix)))) = 1;
%             F_stat_matrix(find(eye(size(F_stat_matrix)))) = 0;
%             
%             % Apply multiple comparison correction (Bonferroni)
%             ntests = ((nLabels^2) - nLabels)/2;
%             pval_matrix_adj = pval_matrix * ntests;
%             pval_matrix_adj(pval_matrix_adj > 1) = 1;
%             pval_matrix = pval_matrix_adj;
%             
%             % Use F-statistic as effect size measure
%             dist_scores_matrix = F_stat_matrix;
%             
%         catch err
%             % Close the progress bar if an error occurs
%             close(h);
%             rethrow(err);
%         end
    end
    % disp('Pairwise MANOVA p-values (upper triangle):');
    % disp(pairwise_p);

    %% PLOTTING 
    % Convert significance matrix to adjacency matrix:
    adjMatrix = (pval_matrix > .01);  % 1 if not significantly different
    adjMatrix(1:size(adjMatrix,1)+1:end) = 1;
    
    % Find connected components:
    G = graph(adjMatrix);
    cluster_labels = conncomp(G);  % assigns cluster number to each node
    num_clusters = max(cluster_labels);
    disp(['Frame' timepoint ' Clustering'])
    disp(cluster_labels)
   
    if plot_confmats == 1
    
        %% 1. Plot distance scores (z-scored)
%         figure;
%         subplot(2,2,1)
%         imagesc(dist_scores_matrix);
%         colorbar;
%         if strcmp(pairwiseTest,'MANOVA')
%             title('Effect size (Wilks Lambda)');
%         else
%             title('Effect size (F statistic, permANOVA)');
%         end
%         xticks(1:nLabels);
%         yticks(1:nLabels);
%         xticklabels(unique_labels);
%         yticklabels(unique_labels);
%         xlabel('Label 1');
%         ylabel('Label 2');
%         axis square;
%         caxis([.20 1]); % focus on low p-values if desired
%     
%         %% 2. plot p-value matrix ---
%         
% %         subplot(2,2,3)
%         figure;
%         imagesc(pval_matrix);
%         colorbar;
%         title('Pairwise P-value Matrix');
%         xticks(1:nLabels);
%         yticks(1:nLabels);
%         xticklabels(unique_labels);
%         yticklabels(unique_labels);
%         xlabel('Label 1');
%         ylabel('Label 2');
%         axis square;
%         caxis([0 .05]); % focus on low p-values if desired
%         colormap(flipud(parula));  % or flipud(hot), flipud(viridis), whatever colormap you are using
%     
%         % Colormap
%         cmap = cbrewer('div','Spectral',length(unique_labels));
%         
        % average for each label
        average_embedding = nan(length(unique_labels),size(embeddings_orig,2));
        for i=1:length(unique_labels)
            idx = find(unique_labels(i)==labels);
            average_embedding(i,:) = mean(embeddings_orig(idx,:),1);
        end
%         
        % Create a mapping from phonemes to clusters based on graph theory
        phoneme_to_cluster = containers.Map();
        for i = 1:length(unique_labels)
            phoneme_to_cluster(char(unique_labels(i))) = cluster_labels(i);
        end
        
        % Create color mapping for all individual data points
        all_clusters = zeros(size(labels));
        for i = 1:length(labels)
            phoneme = labels(i);
            all_clusters(i) = phoneme_to_cluster(char(phoneme));
        end
%         
%         % Display cluster assignments
%         disp('Phoneme graph clustering assignments:');
%         for i = 1:length(unique_labels)
%             phoneme = char(unique_labels(i));
%             cluster = phoneme_to_cluster(phoneme);
%             disp([phoneme, ': Cluster ', num2str(cluster)]);
%         end
%         
        % Add visualization of clusters in the tSNE plot
%         subplot(2,2,2)
        figure;
        % Make a colormap based on graph clusters
        cluster_cmap = parula(num_clusters);
        Y = tsne(embeddings_orig);
        
        % Use graph-based cluster assignments for coloring
        marker_colors = cluster_cmap(all_clusters, :);
        scatter(Y(:,1), Y(:,2), 50, marker_colors, 'filled');
        hold on;
        
        % Add labels
        for i = 1:length(labels)
            text(Y(i,1), Y(i,2), labels(i), 'FontSize', 4);
        end
        
        title(['Graph-based Clusters (', num2str(num_clusters), ' groups)']);
        exportgraphics(gcf, ...
                ['/Volumes/lsa-psyc-djbrang/Freesurfer/LipreadingInTheWild/TCN_LIPREADING_ROOT/Save_Embeddings/500Vis_ClusterPlot_Frame_' timepoint '_' num2str(num_clusters) 'clusters.png'], ...
                'ContentType', 'image');
        exportgraphics(gcf, ...
                ['/Volumes/lsa-psyc-djbrang/Freesurfer/LipreadingInTheWild/TCN_LIPREADING_ROOT/Save_Embeddings/500Vis_ClusterPlot_Frame_' timepoint '_' num2str(num_clusters) 'clusters.eps'], ...
                'ContentType', 'vector');
        close all
        %% 4. t-SNE cluster centers
%         subplot(2,2,4)
        figure;
        Y = tsne(average_embedding);
        
        % Color the centroids by their graph-based cluster assignment
        centroid_colors = zeros(length(unique_labels), 3);
        for i = 1:length(unique_labels)
            cluster_idx = phoneme_to_cluster(char(unique_labels(i)));
            centroid_colors(i,:) = cluster_cmap(cluster_idx,:);
        end
        
        scatter(Y(:,1), Y(:,2), 100, centroid_colors, 'filled');
        hold on;
        title('TSNE Cluster Centers (Graph-based)');
        
        % Add labels to centroids
        for i = 1:length(unique_labels)
            text(Y(i,1), Y(i,2), unique_labels(i), 'FontWeight', 'bold');
        end

%         suptitle(['Frame # relative to speech onset: ',timepoint, ' (', num2str(num_clusters), ' clusters)'])
        exportgraphics(gcf, ...
                ['/Volumes/lsa-psyc-djbrang/Freesurfer/LipreadingInTheWild/TCN_LIPREADING_ROOT/Save_Embeddings/500Vis_ClusterCentroidPlot_Frame_' timepoint '_' num2str(num_clusters) 'clusters.png'], ...
                'ContentType', 'image');
        exportgraphics(gcf, ...
                ['/Volumes/lsa-psyc-djbrang/Freesurfer/LipreadingInTheWild/TCN_LIPREADING_ROOT/Save_Embeddings/500Vis_ClusterCentroidPlot_Frame_' timepoint '_' num2str(num_clusters) 'clusters.eps'], ...
                'ContentType', 'vector');
        close all
    end

    pval_matrix_all(:,:,timex)=pval_matrix;
    dist_scores_matrix_all(:,:,timex)=dist_scores_matrix;


%%
    
    % X: 4500x20 PCA-reduced data
    % labels: 4500x1 categorical or numeric labels (18 unique groups)
    close all
    
    
    % 1. Run t-SNE
    Y = tsne(embeddings, 'NumDimensions', 2, 'Perplexity', 30, 'Standardize', true);
    
    % 2. Plot with ellipses
    groups = unique(labels);
    nGroups = length(groups);
    % colors = lines(nGroups);  % Distinct colors
    colors = cbrewer('div','Spectral',nGroups);
    
    % Reorder labels based on the proximity between groups
    % Ensure matrix is symmetric and square
    % S = cosine similarity matrix (square)
    
    % Convert similarity to a distance matrix
    S = cosine_sim_matrix;
    D = 1 - S;
    
    % Hierarchical clustering (using average linkage)
    Z = linkage(squareform(D), 'average');
    
    % Get optimal leaf order
    order = optimalleaforder(Z, D);
    
    % Reorder the matrix and labels
    S_reordered = S(order, order);
    if ~exist('groups_reordered','var')
        groups_reordered = groups(order);
    end
    % Plot heatmap
    figure;
    imagesc(S_reordered);
    colormap(parula);
    colorbar;
    % axis equal;
    xticks(1:length(groups));
    yticks(1:length(groups));
    xticklabels(groups_reordered);
    yticklabels(groups_reordered);
    xtickangle(45);
    title(['Cosine Similarity Matrix' timepoint]);
    
    exportgraphics(gcf, ...
                ['/Volumes/lsa-psyc-djbrang/Freesurfer/LipreadingInTheWild/TCN_LIPREADING_ROOT/Save_Embeddings/500Vis_reorderedCosine_' timepoint 'clusters_eclipse.png'], ...
                'ContentType', 'image');
    exportgraphics(gcf, ...
            ['/Volumes/lsa-psyc-djbrang/Freesurfer/LipreadingInTheWild/TCN_LIPREADING_ROOT/Save_Embeddings/500Vis_reorderedCosine_' timepoint  'clusters_eclipse.eps'], ...
            'ContentType', 'vector');
    
    
    figure;
    hold on;
    legend_handles = gobjects(nGroups,1);  % Preallocate graphic handles
    for i = 1:nGroups
        idx = labels == groups_reordered(i);
        groupData = Y(idx, :);
    
        % Mean and covariance
        mu = mean(groupData, 1);
        Sigma = cov(groupData);
    
        % Scatter plot (faded)
        scatter(groupData(:,1), groupData(:,2), 10, colors(i,:), 'filled', 'MarkerFaceAlpha', 0.3);
    
        % Ellipse for 1 std deviation
        % plot_gaussian_ellipse(mu, Sigma, 1, colors(i,:), 2);
        legend_handles(i) = plot_gaussian_ellipse(mu, Sigma, 1, colors(i,:), 2);
    
    end
% repeat for labels
    for i = 1:nGroups
        idx = labels == groups_reordered(i);
        groupData = Y(idx, :);
    
        % Mean and covariance
        mu = mean(groupData, 1);
    
        % === Add group label at center of ellipse ===
        text(mu(1), mu(2), sprintf('%s', string(groups_reordered(i))), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'FontSize', 20, ...
            'FontWeight', 'bold', ...
            'Color', 'k');  % You can change text color if needed
    end
    
    xlabel('t-SNE 1');
    ylabel('t-SNE 2');
    title('t-SNE with Group Means and Covariance Ellipses');
    % Set legend only using ellipse handles
    legend(legend_handles,arrayfun(@(x) sprintf('%s', string(x)), groups_reordered, 'UniformOutput', false), ...
           'Location', 'eastoutside');
    axis equal;
    hold off;
    
    exportgraphics(gcf, ...
                ['/Volumes/lsa-psyc-djbrang/Freesurfer/LipreadingInTheWild/TCN_LIPREADING_ROOT/Save_Embeddings/500Vis_ClusterCentroidPlot_Frame_' timepoint 'clusters_eclipse.png'], ...
                'ContentType', 'image');
    exportgraphics(gcf, ...
            ['/Volumes/lsa-psyc-djbrang/Freesurfer/LipreadingInTheWild/TCN_LIPREADING_ROOT/Save_Embeddings/500Vis_ClusterCentroidPlot_Frame_' timepoint  'clusters_eclipse.eps'], ...
            'ContentType', 'vector');
    
    close all;

end



% ===== Function to draw ellipse =====
function h = plot_gaussian_ellipse(mu, Sigma, nsig, color, lw)
    theta = linspace(0, 2*pi, 100);
    circle = [cos(theta); sin(theta)];
    [V, D] = eig(Sigma);
    ellipse = nsig * V * sqrt(D) * circle;
    ellipse = bsxfun(@plus, ellipse, mu');
    h = plot(ellipse(1, :), ellipse(2, :), 'Color', color, 'LineWidth', lw);  % <-- return handle
end

