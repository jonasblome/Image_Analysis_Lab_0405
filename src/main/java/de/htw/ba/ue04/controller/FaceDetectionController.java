/**
 * @author Nico Hezel, Klaus Jung
 */
package de.htw.ba.ue04.controller;

import java.awt.Dimension;
import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import de.htw.ba.facedetection.IntegralImage;
import de.htw.ba.facedetection.StrongClassifier.MatchingResult;
import de.htw.ba.facedetection.WeakClassifier;
import de.htw.ba.ue04.facedetection.FDStrongClassifier;
import de.htw.ba.ue04.facedetection.FDIntegralImage;
import de.htw.ba.ue04.facedetection.FDWeakClassifier;

public class FaceDetectionController extends FaceDetectionBase {

	@Override
	protected void calculateIntegralImage(int[] srcPixels, int width, int height) {
		System.out.println("calculateIntegralImage");
		integralImage = new FDIntegralImage(srcPixels, width, height);
	}

	@Override
	protected void createManualClassifier() {
		System.out.println("createManualClassifier");

		// Average face size to be used for the classifier's detector size 
		Rectangle avgFace = testImage.getAverageFaceDimensions();
		strongClassifier = new FDStrongClassifier(avgFace.width, avgFace.height);

		// Adding weak classifiers to strong classifier
		// Left eye
		Rectangle[] wcRedAreas = new Rectangle[] {new Rectangle(20, 50, 30, 30)};
		Rectangle[] wcGreenAreas = new Rectangle[] {new Rectangle(20, 20, 30, 30)};
		WeakClassifier wc1 = new FDWeakClassifier(wcRedAreas, wcGreenAreas);
		wc1.setWeight(1.0);
		wc1.setThreshold(0.1);
		strongClassifier.addWeakClassifier(wc1);

		// Left eye to hairline
		wcRedAreas = new Rectangle[] {new Rectangle(0, 30, 10, 40)};
		wcGreenAreas = new Rectangle[] {new Rectangle(10, 30, 10, 40)};
		WeakClassifier wc2 = new FDWeakClassifier(wcRedAreas, wcGreenAreas);
		wc2.setWeight(1.0);
		wc1.setThreshold(0.1);
		strongClassifier.addWeakClassifier(wc2);

		// Nose to eyes
		wcRedAreas = new Rectangle[] {new Rectangle(40, 20, 10, 60), new Rectangle(70, 20, 10, 60)};
		wcGreenAreas = new Rectangle[] {new Rectangle(50, 20, 20, 60)};
		WeakClassifier wc3 = new FDWeakClassifier(wcRedAreas, wcGreenAreas);
		wc3.setWeight(1.0);
		wc1.setThreshold(0.1);
		strongClassifier.addWeakClassifier(wc3);

		// Mouth to chin
		wcRedAreas = new Rectangle[] {new Rectangle(40, 110, 40, 10)};
		wcGreenAreas = new Rectangle[] {new Rectangle(40, 100, 40, 10)};
		WeakClassifier wc4 = new FDWeakClassifier(wcRedAreas, wcGreenAreas);
		wc4.setWeight(0.5);
		wc1.setThreshold(0.1);
		strongClassifier.addWeakClassifier(wc4);

		// Forehead to hairline
		wcRedAreas = new Rectangle[] {new Rectangle(20, 0, 80, 10)};
		wcGreenAreas = new Rectangle[] {new Rectangle(20, 10, 80, 30)};
		WeakClassifier wc5 = new FDWeakClassifier(wcRedAreas, wcGreenAreas);
		wc5.setWeight(0.5);
		wc1.setThreshold(0.1);
		strongClassifier.addWeakClassifier(wc5);

		// Normalizing strong classifier weights
		strongClassifier.normalizeWeights();
	}
	
	@Override
	protected void createTrainedClassifier(int weakClassifierCount) {
		System.out.println("createTrainedClassifier");
		
		// Average face size to be used for the classifier's detector size 
		Rectangle avgFace = testImage.getAverageFaceDimensions();
		strongClassifier = new FDStrongClassifier(avgFace.width, avgFace.height);
		ArrayList<WeakClassifier> weakClassifiers = new ArrayList<>();

		Random r = new Random(48);

		// Creating random weak classifiers
		for(int numWC = 0; numWC < weakClassifierCount * 30; numWC++) {
			// Randomizing green areas
			int greenWidth = ThreadLocalRandom.current().nextInt(5, (int) (avgFace.width * 3.0 / 5.0));
			int greenHeight = ThreadLocalRandom.current().nextInt(5, (int) (avgFace.height * 3.0 / 5.0));
			int greenX = ThreadLocalRandom.current().nextInt(0, avgFace.width - greenWidth);
			int greenY = ThreadLocalRandom.current().nextInt(0, avgFace.height - greenHeight);

			// Randomizing red areas
			int redWidth = ThreadLocalRandom.current().nextInt(5, (int) (avgFace.width * 3.0 / 5.0));
			int redHeight = ThreadLocalRandom.current().nextInt(5, (int) (avgFace.height * 3.0 / 5.0));
			int redX = ThreadLocalRandom.current().nextInt(0, avgFace.width - redWidth);
			int redY = ThreadLocalRandom.current().nextInt(0, avgFace.height - redHeight);

			Rectangle[] redAreas = new Rectangle[] {new Rectangle(greenX, greenY, greenWidth, greenHeight)};
			Rectangle[] greenAreas = new Rectangle[] {new Rectangle(redX, redY, redWidth, redHeight)};

			WeakClassifier wC = new FDWeakClassifier(redAreas, greenAreas);
			wC.setWeight(1.0);
			wC.setThreshold(0.05 + r.nextDouble() * (0.15 - 0.05));
			weakClassifiers.add(wC);
		}

		// Training weak classifiers with AdaBoost and constructing a strong classifier with those
		List<Rectangle> faceRectangles = testImage.getFaceRectangles();
		List<Rectangle> nonFaceRectangles = testImage.getNonFaceRectangles();

		// Initializing image weights
		double[] faceWeights = new double[faceRectangles.size()];
		Arrays.fill(faceWeights, 1.0 / (2.0 * faceWeights.length));
		double[] nonFaceWeights = new double[nonFaceRectangles.size()];
		Arrays.fill(nonFaceWeights, 1.0 / (2.0 * nonFaceWeights.length));

		// Training AdaBoost for x iterations
		int iterations = 30;

		for (int i = 0; i < iterations; i++) {
			// Normalizing image weights
			double faceWeightsSum = Arrays.stream(faceWeights).sum();
			faceWeights = Arrays.stream(faceWeights).map(x -> x / faceWeightsSum).toArray();
			double nonFaceWeightsSum = Arrays.stream(nonFaceWeights).sum();
			nonFaceWeights = Arrays.stream(nonFaceWeights).map(x -> x / nonFaceWeightsSum).toArray();

			// Calculating errors for each weak classifier and each picture
			double[][] faceErrors = new double[faceRectangles.size()][weakClassifiers.size()];
			double[][] nonFaceErrors = new double[nonFaceRectangles.size()][weakClassifiers.size()];

			for(int fR = 0; fR < faceRectangles.size(); fR++) {
				for (int wc = 0; wc < weakClassifiers.size(); wc++) {
					faceErrors[fR][wc] = faceWeights[fR] * Math.abs(weakClassifiers.get(wc).matchingAt(integralImage, faceRectangles.get(fR).x, faceRectangles.get(fR).y).featureValue - 1);
				}
			}
			for(int fR = 0; fR < nonFaceRectangles.size(); fR++) {
				for (int wc = 0; wc < weakClassifiers.size(); wc++) {
					nonFaceErrors[fR][wc] = nonFaceWeights[fR] * Math.abs(weakClassifiers.get(wc).matchingAt(integralImage, nonFaceRectangles.get(fR).x, nonFaceRectangles.get(fR).y).featureValue - 0);
				}
			}

			double[] classifierErrorSum = new double[weakClassifiers.size()];
			double lowestClassifierError = Double.POSITIVE_INFINITY;
			int bestClassifier = 0;

			// Accumulating errors for each classifier
			for(int wc = 0; wc < weakClassifiers.size(); wc++) {
				double sum = 0;

				for (int fR = 0; fR < faceRectangles.size(); fR++) {
					sum += faceErrors[fR][wc];
				}
				for (int nFR = 0; nFR < nonFaceRectangles.size(); nFR++) {
					sum += nonFaceErrors[nFR][wc];
				}

				// Setting to small value so that the weight can not result in infinity
				if(sum == 0) {
					sum = 0.001;
				}

				classifierErrorSum[wc] = sum;

				// Choosing best classifier
				if(sum < lowestClassifierError) {
					lowestClassifierError = sum;
					bestClassifier = wc;
				}
			}

			if(lowestClassifierError > 0.5) {
				weakClassifiers.remove(bestClassifier);
			}

			else {
				// Updating image weights
				double beta = classifierErrorSum[bestClassifier] / (1 - classifierErrorSum[bestClassifier]);
				for (int fR = 0; fR < faceRectangles.size(); fR++) {
					int ei = weakClassifiers.get(bestClassifier).matchingAt(integralImage, faceRectangles.get(fR).x, faceRectangles.get(fR).y).featureValue > weakClassifiers.get(bestClassifier).getThreshold() ? 1 : 0;
					faceWeights[fR] = faceWeights[fR] * Math.pow(beta, 1 - ei);
				}

				weakClassifiers.get(bestClassifier).setWeight(Math.log(1 / beta));

				strongClassifier.addWeakClassifier(weakClassifiers.get(bestClassifier));
				weakClassifiers.remove(bestClassifier);
			}
		}

		strongClassifier.normalizeWeights();
	}

    /**
     * Use strongClassifier to calculate a feature heat map.
     * Store all detected regions in the detectionResult list of rectangles.
     * The use of nonMaxSuppression is optional for the exercise.
	 * 
     * @param featureHeatMapPixels
     * @param width
     * @param height
     * @param threshold
     * @param nonMaxSuppression
     */
	protected void doDetection(int[] featureHeatMapPixels, int width, int height, float threshold, boolean nonMaxSuppression) {
		System.out.println("doDetection");
		
	   	// set current threshold for detection
    	strongClassifier.setThreshold(threshold);
     	
     	// detector size
		Dimension size = strongClassifier.getSize();
		
		double featureValue[] = new double[width * height];
		boolean isDetected[] = new boolean[width * height];

		detectionResult = new ArrayList<>();

		// Check strong classifier for all image positions that fully contain the detector region
     	for(int y = 0; y < height - size.height; y++) {	
			for(int x = 0; x < width - size.width; x++)	{
				int pos = y * width + x;
				
				// Calculate feature value and classification result
				MatchingResult result = strongClassifier.matchingAt(integralImage, x, y);
				featureValue[pos] = result.featureValue;
				isDetected[pos] = result.isDetected;
				
				// Draw feature map
				int gray = (int)(featureValue[pos] * 255 / threshold); // increase contrast by inverse threshold
				if(gray < 0)
					gray = 0;
				if(gray > 255)
					gray = 255;
				int red = gray;
				if(isDetected[pos]) {
					// colorize detected positions
					red = 255;
					gray = 0;
					detectionResult.add(new Rectangle(x, y, size.width, size.height));
				}
				
				featureHeatMapPixels[pos] =  (0xFF << 24) | (red << 16) | (gray << 8) | gray;
			}
		}
	}
}
