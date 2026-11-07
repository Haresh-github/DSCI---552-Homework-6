"""Produce a clean, black-text PDF for Homework 6 from verified result files."""
from pathlib import Path
import json, math
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER,TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).parent;FIG=ROOT/'figures'; OUT=ROOT/'results'
orig=json.loads((OUT/'results.json').read_text()); mt=json.loads((OUT/'model_tree_results.json').read_text());cv=pd.read_csv(OUT/'model_tree_cv_summary.csv')
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.pdfbase.pdfmetrics import registerFontFamily
pdfmetrics.registerFontFamily('Times-Roman',normal='Times-Roman',bold='Times-Bold',italic='Times-Italic',boldItalic='Times-BoldItalic')
W,H=letter
pagew=W-112
pdf=ROOT/'DSCI_552_HW6_Report.pdf'
doc=SimpleDocTemplate(str(pdf),pagesize=letter,leftMargin=56,rightMargin=56,topMargin=54,bottomMargin=52,
                      title='APS Failure Classification - DSCI 552 Homework 6',author='Haresh Sambandamoorthy')
styles={
 'Title':ParagraphStyle('title',fontName='Times-Bold',fontSize=20.5,leading=24.0,textColor=colors.black,spaceAfter=2,alignment=TA_CENTER),
 'Author':ParagraphStyle('author',fontName='Times-Roman',fontSize=11.5,leading=16.0,textColor=colors.black,spaceAfter=14,alignment=TA_CENTER),
 'Section':ParagraphStyle('sec',fontName='Times-Bold',fontSize=16.0,leading=20.0,textColor=colors.black,spaceBefore=12,spaceAfter=8,keepWithNext=True),
 'Sub':ParagraphStyle('sub',fontName='Times-Bold',fontSize=14.0,leading=18.2,textColor=colors.black,spaceBefore=10,spaceAfter=6,keepWithNext=True),
 'Body':ParagraphStyle('body',fontName='Times-Roman',fontSize=12.25,leading=17.4,textColor=colors.black,spaceAfter=9.0,firstLineIndent=14,alignment=TA_JUSTIFY),
 'Small':ParagraphStyle('small',fontName='Times-Roman',fontSize=10.55,leading=14.0,textColor=colors.black,spaceAfter=5,alignment=TA_LEFT),
 'Caption':ParagraphStyle('caption',fontName='Times-Italic',fontSize=10.7,leading=14.3,textColor=colors.black,alignment=TA_CENTER,spaceBefore=5,spaceAfter=14),
 'Cell':ParagraphStyle('cell',fontName='Times-Roman',fontSize=9.9,leading=12.8,textColor=colors.black),
 'HeadCell':ParagraphStyle('headcell',fontName='Times-Bold',fontSize=9.9,leading=12.8,textColor=colors.black)}
S=[]
def p(t,style='Body'): S.append(Paragraph(t,styles[style]))
def h(t): p(t,'Section')
def sh(t): p(t,'Sub')
def gap(n=8): S.append(Spacer(1,n))
def pb(): S.append(PageBreak())
def tb(headers,rows,widths=None):
    # Ignore older fixed table widths; scale all proportionally for the new page geometry.
    if widths:
        ratio=pagew/sum(widths)
        widths=[x*ratio for x in widths]
    else:
        widths=[pagew/len(headers)]*len(headers)
    cells=[[Paragraph(str(x),styles['HeadCell']) for x in headers]] + [[Paragraph(str(x),styles['Cell']) for x in row] for row in rows]
    q=Table(cells,colWidths=widths,hAlign='CENTER',repeatRows=1)
    q.setStyle(TableStyle([('LINEBELOW',(0,0),(-1,0),.65,colors.black),
         ('LINEBELOW',(0,-1),(-1,-1),.48,colors.HexColor('#888888')),
         ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f2f2')]),
         ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5.5),
         ('RIGHTPADDING',(0,0),(-1,-1),5.5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    S.append(KeepTogether([q,Spacer(1,10)]))
def pic(filename,caption,max_h=440,max_w=pagew):
    path=FIG/filename
    iw,ih=ImageReader(str(path)).getSize()
    # Reserve comfortable space for nearby explanation and caption.
    actual_limit=min(max_h, 480)
    scale=min(max_w/iw,actual_limit/ih)
    illustration=Image(str(path),width=iw*scale,height=ih*scale)
    caption_flowable=Paragraph(caption,styles['Caption'])
    S.append(KeepTogether([illustration,caption_flowable]))

def pm(m):return f"{100*m:.2f}%"
def auc(m):return f"{m:.4f}"
def perf(name,m):
    a=m['train'];b=m['test'];return [name,pm(b['error']),auc(b['auc']),str(b['fn']),str(b['fp']),pm(b['recall'])]
def bysplit(label,m):
    return [[s.title(),f"{m[s]['tn']:,}",f"{m[s]['fp']:,}",f"{m[s]['fn']:,}",f"{m[s]['tp']:,}",pm(m[s]['error']),auc(m[s]['auc'])] for s in ('train','test')]

def page(canvas,doc):
    canvas.saveState()
    canvas.setFont('Times-Roman',10.5)
    canvas.setFillColor(colors.black)
    canvas.drawCentredString(W/2,32,str(doc.page))
    canvas.restoreState()

p('APS Failure Classification: Tree-Based Methods','Title')
p('Haresh Sambandamoorthy | DSCI 552 | Homework 6','Author')
h('1. Tree-Based Methods')
sh('1(a) APS Failure dataset')
p('The APS data contains 170 numeric readings per truck. The target is whether the air-pressure system has failed. The supplied training and test sets were kept separate.')
d=orig['data']
tb(['Item','Training data','Test data'],[
 ['Rows','60,000','16,000'],['Positive (APS failure)','1,000','375'],['Negative','59,000','15,625'],
 ['Missing feature entries',f"{d['train_missing_cells']:,} ({d['train_missing_pct']:.2f}%)",f"{d['test_missing_cells']:,} ({d['test_missing_pct']:.2f}%)"],
 ['Number of features','170','170']],[pagew*.43,pagew*.28,pagew*.29])
sh('1(b)(i) Handling missing values')
p('Common approaches are deleting incomplete rows, filling blanks with a mean or median, or estimating missing values using nearby rows. Deleting rows would waste too much data here. I used the <b>median of each training feature</b> to fill blanks. The same saved training medians were used for test rows. For cross-validation, medians were recalculated from only the training portion of each fold.')
sh('1(b)(ii) Coefficient of variation')
p('For each of the 170 features, CV = sample standard deviation / mean, using n - 1 in the standard deviation. All values are in <b>results/all_170_feature_cv.csv</b>. Values can be very large when a mean is close to zero. The 13 highest-CV features are:')
p(', '.join(d['cv_top_13']) + '.', 'Small')
p('The 13 features are used for plots only. All 170 features remain available for modeling.')
pic('02_top_13_cv.png','Figure 1. The thirteen highest-CV features; values are based on the training data.',max_h=360)
sh('1(b)(iii) Correlation matrix')
p('The next chart shows correlations across all 170 features. A strong positive or negative correlation means two readings often change together; correlation alone does not prove that a reading is useful for predicting failures.')
pic('01_correlation_all_features.png','Figure 2. Correlation matrix for all 170 features.',max_h=470)
pb()
sh('1(b)(iv) Scatter plots and box plots')
p('The assignment asks for floor(sqrt(170)) = 13 features. For readability, the scatter plots display a fixed sample of 250 positive and 250 negative observations. The box plots use all 60,000 training rows. A log-like plotting scale reduces the visual effect of extremely large values.')
pic('03_scatter_top_13_by_class.png','Figure 3. Scatter plots of the thirteen features with the largest CV.',max_h=425)
pb()
pic('04_boxplots_top_13_by_class.png','Figure 4. Box plots comparing positive and negative classes for all 13 features. Extreme points are hidden from this figure, not from the analysis.',max_h=365)
p('Some plots suggest a difference between failure and non-failure observations, but the plots alone cannot tell us whether a feature is statistically important. High CV may also reflect a few unusually large numbers. We retain every feature in the models.')
pic('05_pairwise_scatter.png','Figure 5. Pairwise scatter plots for five of the thirteen high-CV features.',max_h=410)
sh('1(b)(v) Class counts and imbalance')
p('There are <b>1,000 positives (1.67%)</b> and <b>59,000 negatives (98.33%)</b> in the training data. That is 59 negatives for each positive. This dataset is highly imbalanced: a model predicting only the negative class would appear accurate but would fail to catch any APS failures.')
pic('06_class_balance.png','Figure 6. Training-set class counts.',max_h=210)
sh('Reading the model results')
p('TN means a normal truck correctly called normal; FP means a false alarm; FN means a real failure missed; and TP means a correctly detected failure. Error is the share of wrong predictions. ROC-AUC shows how well the model ranks failures ahead of normal trucks. All confusion matrices use a fixed 0.50 threshold.')
sh('1(c) Random forest without compensation')
p('The forest uses 180 trees and all 170 readings. It does not rebalance the classes.')
tb(['Split','TN','FP','FN','TP','Error','AUC'],bysplit('',orig['models']['random_forest']),[68,70,51,52,54,89,89])
r=orig['models']['random_forest']
p(f"Out-of-bag error = <b>{pm(r['oob_error'])}</b>; test error = <b>{pm(r['test']['error'])}</b>. The test error is {100*(r['test']['error']-r['oob_error']):.2f} percentage points higher. The model misses {r['test']['fn']} of the 375 failures.")
sh('1(d) Random forest with class balancing')
p('A common balancing method is to give rare positive cases more weight. The second forest uses class weights estimated independently for each tree; the remaining forest settings are unchanged.')
tb(['Split','TN','FP','FN','TP','Error','AUC'],bysplit('',orig['models']['weighted_random_forest']),[68,70,51,52,54,89,89])
r2=orig['models']['weighted_random_forest']
p(f"Out-of-bag error = <b>{pm(r2['oob_error'])}</b>; test error = <b>{pm(r2['test']['error'])}</b>. Compared with the unweighted forest, the weighted version misses {r2['test']['fn']-r['test']['fn']} more failures and has {r2['test']['fp']} false alarms. Thus class weighting did not help at the selected decision threshold.")
sh('1(e) and 1(f) Corrected model-tree method')
p('The key correction is to use an actual <b>multivariable split</b> at each internal tree node. That node fits a logistic regression using all 170 input readings, with an L1 penalty to reduce unnecessary coefficients. Its weighted score decides which branch an observation follows. The terminal group reports an estimated failure probability.')
p('To limit unusually large values, the code applies a signed log transformation followed by standard scaling. A tree can grow to depth 3; a proposed split needs enough rows on both sides. No extra class weights or balancing are used in 1(e).')
p('<b>Library distinction:</b> Standard XGBoost does not create these logistic-regression split rules. Its tree splits use individual features and its L1 setting regularizes leaf outputs. Therefore, the primary results in 1(e) and 1(f) are from the custom L1 logistic model tree that implements the described node rule. Conventional XGBoost is also reported separately for comparison, without confusing the two methods.')
sh('Selecting the L1 penalty using five folds')
p('The candidate alpha values are 0.0001, 0.0003, and 0.001. The training data is divided into five balanced folds; each candidate is scored on untouched validation rows. The alpha with the best mean validation AUC is selected. No model setting is chosen using the test set.')
cvrows=[]
for sm in [False,True]:
 for rec in cv[cv.smote==sm].itertuples():
  cvrows.append(['SMOTE' if sm else 'Original',f'{rec.alpha:g}',auc(rec.mean_auc),pm(rec.mean_cv_error),pm(rec.mean_recall)])
tb(['Data','alpha','Validation AUC','Validation error','Failure recall'],cvrows,[78,61,111,119,143])
pic('16_model_tree_alpha_cv.png','Figure 7. Five-fold model-tree validation AUC across the candidate L1 penalties.',max_h=280)
sh('1(e) L1 logistic model tree: original class balance')
plain=mt['models']['l1_logistic_model_tree']
p(f"Cross-validation selects <b>alpha = {plain['alpha']:g}</b>. The final tree contains {plain['internal_nodes']} logistic-regression split nodes. No SMOTE or class weights were used.")
tb(['Split','TN','FP','FN','TP','Error','AUC'],bysplit('',plain),[68,70,51,52,54,89,89])
p(f"Mean five-fold validation error = {pm(plain['cv_error'])}; final test error = {pm(plain['test']['error'])}. Validation AUC = {auc(plain['cv_auc'])}; test AUC = {auc(plain['test']['auc'])}. The model correctly detects {plain['test']['tp']} failures and misses {plain['test']['fn']}.")
sh('1(f) L1 logistic model tree: SMOTE inside each fold')
p('SMOTE creates extra positive examples by combining existing positive rows. <b>Correct order:</b> first split into folds, then learn imputation and scaling from a fold\'s training rows, then apply SMOTE to those training rows only, and finally score untouched validation rows. For the last fit, SMOTE is applied to all 60,000 original training rows, producing 59,000 negative and 59,000 positive training rows. The original test set is never changed.')
sm=mt['models']['l1_logistic_model_tree_smote']
p(f"Cross-validation selects <b>alpha = {sm['alpha']:g}</b>. The final tree contains {sm['internal_nodes']} logistic-regression split nodes.")
tb(['Split','TN','FP','FN','TP','Error','AUC'],bysplit('',sm),[68,70,51,52,54,89,89])
p(f"Mean validation error = {pm(sm['cv_error'])}; final test error = {pm(sm['test']['error'])}. Validation AUC = {auc(sm['cv_auc'])}; test AUC = {auc(sm['test']['auc'])}.")
p(f"<b>Comparison:</b> SMOTE raises caught failures from {plain['test']['tp']} to {sm['test']['tp']} and reduces missed failures from {plain['test']['fn']} to {sm['test']['fn']}. False alarms increase from {plain['test']['fp']} to {sm['test']['fp']}, so its total error is higher despite detecting more failures. This trade-off matters in real fault detection.")
pb()
pic('17_model_tree_roc.png','Figure 8. Training and test ROC curves for the corrected L1 logistic model trees.',max_h=305)
pic('18_model_tree_confusion.png','Figure 9. Confusion matrices for both model trees on training and test data.',max_h=355)
sh('Additional comparison: conventional XGBoost')
p('The following numbers come from an ordinary XGBoost tree classifier with L1-regularized leaf weights (not logistic-regression split nodes). Five-fold validation chose its L1 settings from 0.1, 1, and 10. This comparison is included because the handout also names XGBoost, but its results are <b>not</b> used as a substitute for the requested model-tree calculations.')
tb(['Model','Train AUC','Test AUC','Test error','Missed','False alarms'],[
 ['Standard XGBoost',auc(orig['models']['xgboost']['train']['auc']),auc(orig['models']['xgboost']['test']['auc']),pm(orig['models']['xgboost']['test']['error']),orig['models']['xgboost']['test']['fn'],orig['models']['xgboost']['test']['fp']],
 ['XGBoost + SMOTE',auc(orig['models']['xgboost_smote']['train']['auc']),auc(orig['models']['xgboost_smote']['test']['auc']),pm(orig['models']['xgboost_smote']['test']['error']),orig['models']['xgboost_smote']['test']['fn'],orig['models']['xgboost_smote']['test']['fp']]
 ],[124,72,82,82,77,75])
p(f"The conventional XGBoost runs selected alpha = {orig['models']['xgboost']['alpha']} without SMOTE and alpha = {orig['models']['xgboost_smote']['alpha']} with SMOTE. Their train/test confusion matrices and ROC curves, five-fold validation results, and executable code are retained in the package as supplementary evidence.")
pic('07_roc_curves.png','Figure 10. ROC curves for the original forest and conventional XGBoost comparisons.',max_h=340)
pb()
sh('Summary of the test results')
comparison=[perf('Random forest',orig['models']['random_forest']),perf('Weighted forest',orig['models']['weighted_random_forest']),perf('L1 logistic model tree',plain),perf('L1 model tree + SMOTE',sm),perf('Conventional XGBoost',orig['models']['xgboost']),perf('XGBoost + SMOTE (extra)',orig['models']['xgboost_smote'])]
tb(['Model','Test error','AUC','Missed','False alarms','Recall'],comparison,[152,77,62,66,76,79])
p('The custom logistic model tree matches the requested split structure. Conventional XGBoost scores higher on AUC here, but its split structure is different. The results should not be interpreted as identical methods.')
p('For an optional illustration of the APS costs in the source description, false alarms cost 10 units each and missed failures cost 500 units. Under that rule, the two L1 model trees cost '+f"{plain['test']['cost']:,} and {sm['test']['cost']:,} units, respectively. This cost was <b>not</b> used for tuning the models.")
h('2. ISLR 6.6.3 - Lasso constraint')
p('The allowed total absolute size of the coefficients is increased (s gets larger), giving the model more freedom.')
tb(['Quantity','Choice','Why'],[
 ['(a) Training RSS','iv - Decreases','A model with more freedom can fit training data at least as well, or equally well.'],
 ['(b) Test RSS','ii - U-shaped','Some flexibility helps at first; too much may fit noise.'],
 ['(c) Variance','iii - Increases','A flexible fit can change more when the training data changes.'],
 ['(d) Squared bias','iv - Decreases','The fit can capture more of the true pattern.'],
 ['(e) Irreducible error','v - Constant','Unpredictable noise is not changed by s.']],[135,127,250])
p('These are the usual broad patterns intended by the exercise; a single real dataset may behave differently.')
pb()
h('3. ISLR 6.6.5 - Identical predictors')
p('Let the two predictors be identical, taking the values a and -a across the two observations, with responses b and -b. Define t = beta1 + beta2. The fitted value depends only on t, not on the separate coefficients.')
sh('(a) Ridge objective')
p('Minimize 2(b - a t)<super>2</super> + lambda(beta1<super>2</super> + beta2<super>2</super>).')
sh('(b) Ridge coefficients')
p('For any fixed t, the squared penalty is smallest when beta1 = beta2 = t/2. The resulting ridge estimates are beta1 = beta2 = 2ab/(4a<super>2</super> + lambda), for positive lambda.')
sh('(c) Lasso objective')
p('Minimize 2(b - a t)<super>2</super> + lambda(|beta1| + |beta2|).')
sh('(d) Why the lasso answers need not be unique')
p('When the two coefficients have the same sign, their absolute values add to |t|. Thus (0,t), (t/2,t/2), and (t,0) all give the same predictions and penalty. The best total is sign(b/a) times max(|b/a| - lambda/(4a<super>2</super>), 0), when a is nonzero. If this total is nonzero, there are infinitely many equally good coefficient pairs. If it is zero with positive lambda, (0,0) is the unique pair.')
h('4. ISLR 8.4.5 - Bagging: votes versus probabilities')
p('Six of the ten trees predict the red class with probability above 0.50, while four predict below 0.50. <b>Majority voting predicts Red.</b> The ten probabilities add to 4.50, giving an average of 0.45. <b>Averaging probabilities predicts Green.</b> Voting counts the winners only, while averaging uses the confidence of every tree.')
pb()
h('5. ISLR 9.7.3 - Maximum-margin separator')
pic('11_islr_9_7_3_diagram.png','Figure 11. Maximum-margin line (solid), two margins (dashed), alternative line and a new point.',max_h=260)
p('(a) Red points: (3,4), (2,2), (4,4), (1,4). Blue points: (2,1), (4,3), (4,1).')
p('(b) The maximum-margin separating line is X2 = X1 - 0.5, or 0.5 - X1 + X2 = 0.')
p('(c) Predict Red when 0.5 - X1 + X2 is positive, and Blue otherwise. One valid coefficient set is beta0 = 0.5, beta1 = -1, beta2 = 1.')
p('(d) The two margin boundaries are X2 = X1 and X2 = X1 - 1. Each is 0.5/sqrt(2) (about 0.354) from the separating line.')
p('(e) Support vectors: Red points 2 and 3; Blue points 5 and 6. They lie on the two margin boundaries.')
p('(f) Moving point 7 slightly will not change the maximum-margin line, because it does not touch either margin.')
p('(g) The line X2 = 0.75 X1 + 0.30 also separates the original groups but has a smaller margin, so it is not optimal.')
p('(h) Add a Red point at (3,2). This falls on the segment between two Blue points, so no straight line can then separate all Red points from all Blue points.')
pb()
h('6. Extra practice (optional)')
sh('ISLR 5.4.2 - Bootstrap inclusion chance')
p('(a)-(c) Any one draw misses observation j with probability (n-1)/n. All n independent draws miss it with probability (1-1/n)<super>n</super>; therefore its probability of appearing at least once is 1 - (1-1/n)<super>n</super>.')
p('(d) n = 5: 0.67232. (e) n = 100: 0.63397. (f) n = 10,000: 0.63214. (g) The value approaches 1-1/e = 0.63212 as n grows. (h) A simulation of 10,000 bootstrap samples of size 100 produced a proportion of 0.6304, close to the theoretical 0.6340.')
pic('12_bootstrap_probability.png','Figure 12. Probability that an observation appears at least once in a bootstrap sample.',max_h=260)
sh('ISLR 6.8.4 (first-edition numbering) - Ridge as lambda increases')
p('(a) Training RSS: iii, increases. (b) Test RSS: ii, usually U-shaped. (c) Variance: iv, decreases. (d) Squared bias: iii, increases. (e) Irreducible error: v, remains constant.')
pb()
sh('ISLR 8.4.4 - Tree partitions')
p('(a) The pictured split puts X1 at least 1 in a region with prediction 5. On the left, a cut at X1 = 0 and then X2 = 1 gives the four remaining terminal values 15, 3, 0, and 10. (b) For the provided tree, first split at X2 = 1. For X2 below 1, split on X1 = 1 into values -1.80 and 0.63. For X2 between 1 and 2, split on X1 = 0 into -1.06 and 0.21. At X2 at least 2, predict 2.49.')
pic('13_islr_8_4_4_partitions.png','Figure 13. Regions corresponding to the two decision-tree questions.',max_h=280)
pic('15_islr_8_4_4_tree.png','Figure 14. Tree matching the first partition diagram.',max_h=210)
pb()
sh('ISLR 9.7.2 - Nonlinear decision boundary')
p('(a) (1 + X1)<super>2</super> + (2 - X2)<super>2</super> = 4 is a circle with center (-1,2) and radius 2. (b) Values above 4 are outside; values at most 4 are inside or on the circle. (c) (0,0) Blue; (-1,1) Red; (2,2) Blue; (3,8) Blue. (d) Expanding gives 1 + 2 X1 + X1<super>2</super> - 4 X2 + X2<super>2</super> = 0, a linear rule in the expanded inputs X1, X1<super>2</super>, X2 and X2<super>2</super>.')
pic('14_islr_9_7_2_circle.png','Figure 15. The circular boundary and four example observations.',max_h=225)
h('References')
p('[1] DSCI 552 Homework 6, user-supplied question sheet, and the accompanying APS training and test files.','Small')
p('[2] UCI Machine Learning Repository: APS Failure at Scania Trucks. https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks','Small')
p('[3] James, Witten, Hastie and Tibshirani. An Introduction to Statistical Learning, first and second editions. https://www.statlearning.com/','Small')
p('[4] XGBoost documentation, Parameters and Introduction to Boosted Trees: https://xgboost.readthedocs.io/en/stable/parameter.html and https://xgboost.readthedocs.io/en/stable/tutorials/model.html','Small')
p('[5] scikit-learn documentation, SGDClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html','Small')
h('AI use acknowledgment')
p('AI tools were used to assist with organizing the code, compiling the report, and checking formatting. All reported numerical results were generated from the included APS datasets using the accompanying reproducible code.')

# Retain a separate compact report of textbook exercises 2 through 6.
# Both PDFs draw from the same answer flowables; nothing is re-answered.
all_story=S[:]
start_islr=next(i for i,flowable in enumerate(all_story) if isinstance(flowable,Paragraph) and flowable.text.startswith('2. ISLR 6.6.3'))
islr_story=[Paragraph('ISLR Exercises: Homework 6',styles['Title']), Paragraph('Haresh Sambandamoorthy | DSCI 552 | Homework 6',styles['Author'])]+all_story[start_islr:]
# Remove the leading page break if a previous section added one.
from reportlab.platypus import PageBreak as PBreak
while isinstance(islr_story[2],PBreak): islr_story.pop(2)
doc.build(S,onFirstPage=page,onLaterPages=page)
extra_pdf=ROOT/'DSCI_552_HW6_ISLR_Solutions.pdf'
extra_doc=SimpleDocTemplate(str(extra_pdf),pagesize=letter,leftMargin=56,rightMargin=56,topMargin=54,bottomMargin=52,
                            title='ISLR Exercises - DSCI 552 Homework 6',author='Haresh Sambandamoorthy')
extra_doc.build(islr_story,onFirstPage=page,onLaterPages=page)
print('Wrote',extra_pdf,extra_pdf.stat().st_size)
print('Wrote',pdf,pdf.stat().st_size)
