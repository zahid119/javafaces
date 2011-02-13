package facerecognition.utils;

import java.util.Comparator;

public class ValueIndexPairComparator implements Comparator<ValueIndexPair>{
	@Override
	public int compare(ValueIndexPair dp1, ValueIndexPair dp2) {
		return new Double(dp2.getVectorElement()).compareTo(new Double(dp1.getVectorElement()));
	}

	
}
