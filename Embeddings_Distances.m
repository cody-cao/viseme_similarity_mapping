% MATLAB script to test if two (or more) label groups are significantly different
% based on full pairwise distances
clear
clc
close all

if ismac
    homeDir = '/Volumes/lsa-psyc-djbrang/Freesurfer/';
else
    homeDir = '/nfs/turbo/lsa-psyc-djbrang/Freesurfer/';
end
pairwiseTest = 'MANOVA'; %'permANOVA'
addpath([homeDir 'LipreadingInTheWild/Fathom/'])

% Cody
projDir = [homeDir '/LipreadingInTheWild/TCN_LIPREADING_ROOT/Save_Embeddings/'];

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
data_dir= [projDir '/train_logs/tcn/2025-04-29T182603'];


% load in the csv that has phoneme labels. use that instead of labels
% For 18 stim, recode
recoded_labels = repmat(["B","M","N","P","S","W"],150,1); % 3 words x 50 each
recoded_labels = recoded_labels(:);
rm_hits = [];


% % 500 vis. soft zeroing 12-29 . speech onset at 11
% data_dir = '/Users/djbrang/University of Michigan Dropbox/David Brang/MAIN/PROJECTS/DeepLearning_TCN_LIPREADING_ROOT/Save_Embeddings/train_logs/tcn/2025-04-28T133008';

% 500 vis. center on 4th frame ([3]). train to 8 epochs
% data_dir = '/Users/djbrang/University of Michigan Dropbox/David Brang/MAIN/PROJECTS/DeepLearning_TCN_LIPREADING_ROOT/Save_Embeddings/train_logs/tcn/2025-04-30T084159';

% 500 vis. center on 4th frame ([3]). train to 8 epochs. soft zeroing 6-29 [5:29]
% data_dir = '/Users/djbrang/University of Michigan Dropbox/David Brang/MAIN/PROJECTS/DeepLearning_TCN_LIPREADING_ROOT/Save_Embeddings/train_logs/tcn/2025-04-30T091759';

% 500 aud. center on 4th frame ([3]). train to 14 epochs
% data_dir = '/Users/djbrang/University of Michigan Dropbox/David Brang/MAIN/PROJECTS/DeepLearning_TCN_LIPREADING_ROOT/Save_Embeddings/train_logs/tcn/2025-04-30T081927';




% % 500 from Keyword_Phonemes
% recoded_labels = {'a'	'a'	'a'	'a'	'a'	'ah'	'ah'	'ah'	'ah'	'ah'	'ai'	'aw'	'aw'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'b'	'd'	'd'	'd'	'd'	'd'	'd'	'd'	'd'	'd'	'd'	'd'	'd'	'd'	'd'	'd'	'dh'	'dh'	'dh'	'dh'	'dh'	'dzh'	'dzh'	'dzh'	'dzh'	'dzh'	'dzh'	'E'	'E'	'E'	'E'	'E'	'E'	'E'	'E'	'E'	'E'	'E'	'E'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'f'	'g'	'g'	'g'	'g'	'g'	'g'	'g'	'g'	'g'	'g'	'g'	'g'	'g'	'g'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'h'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'i'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'I'	'j'	'j'	'j'	'j'	'j'	'j'	'j'	'j'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'k'	'kw'	'kw'	'kw'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'l'	'long schwa'	'long_i'	'long_i'	'long_i'	'long_i'	'longQ'	'longQ'	'longQ'	'longQ'	'longQ'	'longQ'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'm'	'n'	'n'	'n'	'n'	'n'	'n'	'n'	'n'	'n'	'n'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'p'	'Q'	'Q'	'Q'	'Q'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	'r'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	's'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'schwa'	'sh'	'sh'	't'	't'	't'	't'	't'	't'	't'	't'	't'	't'	't'	't'	't'	't'	'th'	'th'	'th'	'th'	'th'	'th'	'th'	'th'	'th'	'tsh'	'tsh'	'tsh'	'tsh'	'tsh'	'tsh'	'tsh'	'tsh'	'tsh'	'tsh'	'tsh'	'tsh'	'v'	'v'	'v'	'v'	'v'	'v'	'v'	'v'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'	'w'};
% recoded_labels = repmat(string(recoded_labels),50,1);
% recoded_labels = recoded_labels(:);
% 
% % Remove vowells - quick approach
% unique_labels = unique(recoded_labels);
% rm_idx = [1 2 4 5 6 7 15 20:22 28];
% rm_hits = [];
% for i=rm_idx
%     rm_hits = [rm_hits;find(recoded_labels==unique_labels(i))];
% end
% recoded_labels(rm_hits) = [];





% time_array = arrayfun(@(x) sprintf('%+d', x), -10:10, 'UniformOutput', false);
time_array = arrayfun(@(x) sprintf('%+d', x), -3:0, 'UniformOutput', false);
plot_confmats = 1;

pval_matrix_all = [];
dist_scores_matrix_all = [];

for timex = 1:4%1:length(time_array)

    % Optimized MATLAB script to test if label groups are significantly different
    % based on full pairwise distances without nested loops, now for all class pairs

    % --- Load the CSV ---
    cd(data_dir)
    timepoint = time_array{timex};
    warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');
    data = readtable(['stimulus_embeddings__offset',timepoint,'.csv'], 'ReadVariableNames', true, 'VariableNamingRule', 'modify');
    labels = string(data.Stimulus);        % force labels into string type
    embeddings = table2array(data(:,4:end));

    
    % Use grouped labels
    labels = recoded_labels;
    embeddings(rm_hits,:)=[];


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
    [coeff, score, latent] = pca(embeddings);

    % Keep top k components
    k = 20; % or 10
    embeddings = score(:,1:k);  % Now 100 x k instead of 100 x 512

    % embeddings: 100x512 matrix
    % labels: 100x1 vector (e.g., 0 and 1)

    % Run MANOVA Globally
    [G, groupNames] = findgroups(labels);
    [d, p_global, stats] = manova1(embeddings, labels);
    % disp(['Global MANOVA p-value: ', num2str(p_global)]);

    unique_labels = unique(labels);
    nLabels = numel(unique_labels);

    % Number of tests to correct
    ntests=((nLabels^2)-nLabels)/2;

    pval_matrix = NaN(nLabels);
    dist_scores_matrix= NaN(nLabels);

    for i = 1:nLabels
        for j = i+1:nLabels
            % Correct way to subset with string labels
            idx = ismember(labels, [unique_labels(i), unique_labels(j)]);

            emb_sub = embeddings(idx,:);
            labels_sub = labels(idx);

            [~, p, stats] = manova1(emb_sub, labels_sub);
            pval_matrix(i,j) = p;

            % Wilks' lambda:
            eigenvalues = stats.eigenval;
            dist_scores_matrix(i,j) = prod(1 ./ (1 + eigenvalues));
           
        end
    end

    % disp('Pairwise MANOVA p-values (upper triangle):');
    % disp(pairwise_p);






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
            pval_matrix(j,i) = pval_matrix(i,j);  % Copy upper triangle to lower triangle
            dist_scores_matrix(j,i) = dist_scores_matrix(i,j);  % Copy upper triangle to lower triangle
        end
    end
    % Fill in diagonal
    pval_matrix(find(eye(size(pval_matrix))))=1;
    dist_scores_matrix(find(eye(size(dist_scores_matrix))))=1;



    if plot_confmats ==1
        % Plot distance scores (z-scored)
        figure('Position',[600 100 1000 1000]);
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
        caxis([.20 1]); % focus on low p-values if desired



        % plot p-value matrix ---
        
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
        caxis([0 .05]); % focus on low p-values if desired
        colormap(flipud(parula));  % or flipud(hot), flipud(viridis), whatever colormap you are using


        % Colormap
        cmap = cbrewer('div','Spectral',length(unique_labels));
        % cmap_all = [];
        % for i=1:length(labels)
        %     idx = find(labels(i)==unique_labels);
        %     cmap_all(i,:)= cmap(idx,:);
        % end

        % [idx, C] = kmeans(embeddings, clusters);
        % Visualize with t-SNE
        subplot(2,2,2)
        % cmap = cbrewer('qual','Set1',length(unique_labels));
        Y = tsne(embeddings_orig);
        gscatter(Y(:,1), Y(:,2), labels,cmap);
        hold on;
        title(['TSNE Clusters']);

        % average for each label
        average_embedding = nan(length(unique_labels),size(embeddings_orig,2));
        for i=1:length(unique_labels)
            idx = find(unique_labels(i)==labels);
            average_embedding(i,:) = mean(embeddings_orig(idx,:),1);
        end
        subplot(2,2,4)
        Y = tsne(average_embedding);
        gscatter(Y(:,1), Y(:,2), unique_labels,cmap);
        hold on;
        title(['TSNE Cluster Centers']);
        text(Y(:,1), Y(:,2), unique_labels);  % label each point
        suptitle(['Frame # relative to speech onset: ',timepoint])
        
        print(gcf,['/Volumes/lsa-psyc-djbrang/Freesurfer/LipreadingInTheWild/TCN_LIPREADING_ROOT/Save_Embeddings/' 'Frame: ' timepoint,'.eps'],"ContentType","vector")
        close all
    end

    pval_matrix_all(:,:,timex)=pval_matrix;
    dist_scores_matrix_all(:,:,timex)=dist_scores_matrix;

end


