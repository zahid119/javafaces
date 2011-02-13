package facerecognition.utils;


public class ValueIndexPair{
	double vectorElement;
	int matrixRowIndex;
	public ValueIndexPair() {
		super();
	}
	public ValueIndexPair(double vectorElement, int matrixRowIndex) {
		super();
		this.vectorElement = vectorElement;
		this.matrixRowIndex = matrixRowIndex;
	}
	public double getVectorElement() {
		return vectorElement;
	}
	public void setVectorElement(double vectorElement) {
		this.vectorElement = vectorElement;
	}
	public int getMatrixRowIndex() {
		return matrixRowIndex;
	}
	public void setMatrixRowIndex(int matrixRowIndex) {
		this.matrixRowIndex = matrixRowIndex;
	}
	
}

