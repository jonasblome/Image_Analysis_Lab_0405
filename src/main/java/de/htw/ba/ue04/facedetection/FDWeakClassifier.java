package de.htw.ba.ue04.facedetection;

import de.htw.ba.facedetection.IntegralImage;
import de.htw.ba.facedetection.WeakClassifier;

import java.awt.*;

public class FDWeakClassifier implements WeakClassifier {
    private Rectangle[] redAreas;
    private Rectangle[] greenAreas;
    private double threshold;
    private double weight;

    public FDWeakClassifier(Rectangle[] redAreas, Rectangle[] greenAreas) {
        this.redAreas = redAreas;
        this.greenAreas = greenAreas;
    }

    @Override
    public Rectangle getPositionInDetector() {
        return null;
    }

    @Override
    public WeakMatchingResult matchingAt(IntegralImage image, int x, int y) {
        double featureValue = 0;

        for (Rectangle greenArea: greenAreas) {
            featureValue += image.meanValue(greenArea.x + x, greenArea.y + y, greenArea.width, greenArea.height) / (255 * greenAreas.length);
        }
        for (Rectangle redArea: redAreas) {
            featureValue -= image.meanValue(redArea.x + x, redArea.y + y, redArea.width, redArea.height) / (255 * redAreas.length);
        }

        WeakMatchingResult result = new WeakMatchingResult();
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
    public void setWeight(double weight) {
        this.weight = weight;
    }

    @Override
    public double getWeight() {
        return weight;
    }

    @Override
    public void drawAt(Graphics2D g2d, int x, int y) {
        g2d.setColor(new Color(0.0f, 1.0f, 0.0f, 0.5f));
        for (Rectangle greenArea: greenAreas) {
            g2d.fill(greenArea);
        }
        g2d.setColor(new Color(1.0f, 0.0f, 0.0f, 0.5f));
        for (Rectangle redArea: redAreas) {
            g2d.fill(redArea);
        }
    }
}
