package de.htw.ba.ue04.facedetection;

import java.awt.Dimension;
import java.awt.Graphics2D;
import java.util.ArrayList;

import de.htw.ba.facedetection.IntegralImage;
import de.htw.ba.facedetection.StrongClassifier;
import de.htw.ba.facedetection.WeakClassifier;

public class FDStrongClassifier implements StrongClassifier {

	private Dimension size;
	public ArrayList<WeakClassifier> weakClassifiers;
	double threshold;
	
	public FDStrongClassifier(int width, int height) {
		size = new Dimension(width, height);
		weakClassifiers = new ArrayList<>();
	}

	@Override
	public void setSize(Dimension size) {
		this.size = size;
	}

	@Override
	public Dimension getSize() {
		return size;
	}

	@Override
	public void addWeakClassifier(WeakClassifier classifier) {
		weakClassifiers.add(classifier);
	}

	@Override
	public void normalizeWeights() {
		// Normalizing weights of all weak classifiers
		double sumOfWeights = 0;

		for (WeakClassifier weakClassifier: weakClassifiers) {
			sumOfWeights += weakClassifier.getWeight();
		}
		for (WeakClassifier weakClassifier: weakClassifiers) {
			weakClassifier.setWeight(weakClassifier.getWeight() / sumOfWeights);
		}
	}

	@Override
	public MatchingResult matchingAt(IntegralImage image, int x, int y) {
		double featureValue = 0;

		// Sum up the weight * featureValue of all weak classifiers
		for (WeakClassifier weakClassifier:	weakClassifiers) {
			featureValue += weakClassifier.matchingAt(image, x, y).featureValue * weakClassifier.getWeight();
		}

		MatchingResult result = new MatchingResult();
		result.featureValue = featureValue;
		result.isDetected = featureValue >= threshold;
		return result;
	}

	@Override
	public void setThreshold(double threshold) {
		this.threshold = threshold;
	}

	@Override
	public double getThreshold() {
		return threshold;
	}

	@Override
	public void drawAt(Graphics2D g2d, int x, int y) {
		// Draw red and green areas of each weak classifier
		for (WeakClassifier weakClassifier: weakClassifiers) {
			weakClassifier.drawAt(g2d, x, y);
		}
	}

}
