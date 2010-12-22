package facerecognition.javafaces;

import java.awt.image.*;
import java.io.*;
import java.util.*;
import javax.imageio.ImageIO;


public class FaceRec{
	private FaceBundle bundle;
	private double[][] weights;	
	public MatchResult findMatchResult(String imageFileName,int selectedeigenfaces,double thresholdVal){
		boolean match = false;
		String message = null;
		String matchingFileName = "";
		double minimumDistance = 0.0;
		try{
			checkImageSizeCompatibility(imageFileName);
			Matrix2D inputFace = getNormalisedInputFace(imageFileName);				
			inputFace.subtract(new Matrix2D(bundle.getAvgFace(),1));
			Matrix2D inputWts = getInputWeights(selectedeigenfaces, inputFace);				
			double[] distances = getDistances(inputWts);		
			ImageDistanceInfo distanceInfo = getMinimumDistanceInfo(distances);
			minimumDistance = Math.sqrt(distanceInfo.getValue());
			matchingFileName = getMatchingFileName(distanceInfo);
			
			if (minimumDistance > thresholdVal){
				message="no match found, try higher threshold";
			}else{
				match = true;
				message = "matching image found";
			}
		}catch(Exception e){
			return new MatchResult(false,"",Double.NaN,e.getMessage());
		}
		return new MatchResult(match,matchingFileName,minimumDistance,message);		
	}
	
	private void checkImageSizeCompatibility(String fileName) throws IOException, FaceRecError{
		BufferedImage image = ImageIO.read(new File(fileName));
		int height = image.getHeight();
		int width = image.getWidth();
		
		int facebundleWidth = this.bundle.getImageWidth();
		int facebundleHeight = this.bundle.getImageHeight();
		if ((height != facebundleHeight) || (width != facebundleWidth)  ){
			throw new FaceRecError("selected image dimensions does not match dimensions of other images");
		}
			
	}
	
	private String getMatchingFileName(ImageDistanceInfo distanceInfo) {
		List<String> imageFileNames = this.bundle.getImageFileNamesList();
		String matchingFileName = imageFileNames.get(distanceInfo.getIndex());
		return matchingFileName;
	}
	
	private Matrix2D getInputWeights(int selectedeigenfaces, Matrix2D inputFace) {
		double[][] eigenFacesArray = this.bundle.getEigenFaces();
		Matrix2D eigenFacesMatrix=new Matrix2D(eigenFacesArray);
		Matrix2D eigenFacesMatrixPart=eigenFacesMatrix.getSubMatrix(selectedeigenfaces);
		Matrix2D eigenFacesMatrixPartTranspose=eigenFacesMatrixPart.transpose();
		Matrix2D inputWts=inputFace.multiply(eigenFacesMatrixPartTranspose);
		return inputWts;
	}
	
	private Matrix2D getNormalisedInputFace(String imageFileName) throws FaceRecError {
		double[] inputFaceData = getImageData(imageFileName);
		Matrix2D inputFace=new Matrix2D(inputFaceData,1);		
		inputFace.normalise();
		return inputFace;
	}

	private ImageDistanceInfo getMinimumDistanceInfo(double[] distances) {
		double minimumDistance = Double.MAX_VALUE;
		int index=0;
		for (int i = 0; i < distances.length; i++) {
			if (distances[i] < minimumDistance) { 
				minimumDistance=distances[i];
				index=i;				
			}
		}
		return new ImageDistanceInfo(distances[index], index);		
	}
	
	private double[] getDistances(Matrix2D inputWt) {			
		Matrix2D tempWt=new Matrix2D(this.weights);	
		double[] inputWtData=inputWt.flatten();		
		tempWt.subtractFromEachRow(inputWtData);
		tempWt.multiplyElementWise(tempWt);
		double[][] temp=tempWt.toArray();
		double[] distances = new double[temp.length];
		for (int i = 0; i < temp.length; i++) {
			double sum = 0.0;
			for (int j = 0; j < temp[0].length; j++) {
				sum += temp[i][j];
			}
			distances[i] = sum;
		}
		return distances;
	}
	
	private double[] getImageData(String imageFileName) throws FaceRecError {
		BufferedImage bi =null;
		double[] inputFace =null;
		try{
			bi = ImageIO.read(new File(imageFileName));
		}catch(IOException ioe){
			throw new FaceRecError(ioe.getMessage());
		}
		if (bi!=null){
		int imageWidth = bi.getWidth();
		int imageHeight = bi.getHeight();
		inputFace = new double[imageWidth * imageHeight];
		bi.getData().getPixels(0, 0, imageWidth, imageHeight,inputFace);
		}
		return inputFace;
	}	
	
	private void doCalculations(String dir,List<String> imglist,int selectedNumOfEigenFaces) throws FaceRecError, IOException{
		FaceBundle b=createFaceBundle(imglist);		
		double[][] wts=calculateWeights(b,selectedNumOfEigenFaces);		
		this.bundle=b;
		this.weights=wts;
		writeCache(dir,b);
	}
	
	private double[][] calculateWeights(FaceBundle b,int selectedNumOfEigenFaces){
		Matrix2D eigenFaces=new Matrix2D(b.getEigenFaces());
		Matrix2D eigenFacesPart=eigenFaces.getSubMatrix(selectedNumOfEigenFaces);
		Matrix2D adjustedFaces=new Matrix2D(b.getAdjustedFaces());
		Matrix2D eigenFacesPartTr=eigenFacesPart.transpose();
		Matrix2D wts=adjustedFaces.multiply(eigenFacesPartTr);
		return wts.toArray();
	}
	
	public FaceBundle createFaceBundle(List<String> filenames) throws FaceRecError, IOException{
		BufferedImage[] bufimgs=getGrayScaleImages(filenames);
		checkImageDimensions(filenames, bufimgs);
		Matrix2D imagesData=getNormalisedImagesData(bufimgs);
		double[] averageFace=imagesData.getAverageOfEachColumn();
		imagesData.adjustToZeroMean();
		EigenvalueDecomposition egdecomp = getEigenvalueDecomposition(imagesData);
		double[] eigenvalues=egdecomp.getEigenValues();
		double[][]eigvectors=egdecomp.getEigenVectors();
		sortEigenVectors(eigenvalues,eigvectors);
		Matrix2D eigenFaces=getNormalisedEigenFaces(imagesData,new Matrix2D(eigvectors));
		int imageWidth=bufimgs[0].getWidth();
		createEigenFaceImages(eigenFaces,imageWidth);
		int imageHeight=bufimgs[0].getHeight();
		FaceBundle b = new FaceBundle(filenames,imagesData.toArray(),averageFace,eigenFaces.toArray(),eigenvalues,imageWidth,imageHeight);
		return b;	
	}
	
	private EigenvalueDecomposition getEigenvalueDecomposition(
			Matrix2D imagesData) {
		Matrix2D imagesDataTr=imagesData.transpose();
		Matrix2D covarianceMatrix=imagesData.multiply(imagesDataTr);
		EigenvalueDecomposition egdecomp =covarianceMatrix.getEigenvalueDecomposition();
		return egdecomp;
	}
	
	public void createEigenFaceImages(Matrix2D egfaces,int imgwidth) throws IOException{
		double[][] eigenfacesArray=egfaces.toArray();						
		String fldrname=".."+File.separator+"eigenfaces";
		makeNewFolder(fldrname);
		String prefix="eigen";
		String ext=".png";		
		for(int i=0;i<eigenfacesArray.length;i++){
			double[] egface=eigenfacesArray[i];
			String filename=fldrname+File.separator+prefix+i+ext;			
			createImageFromArray(filename,egface,imgwidth);
		}
	}
	
	private Matrix2D getNormalisedEigenFaces(Matrix2D imagesData,Matrix2D eigenVectors) {
		Matrix2D eigenVectorsTr=eigenVectors.transpose();
		Matrix2D eigenFaces=eigenVectorsTr.multiply(imagesData);
		double[][] eigenFacesData=eigenFaces.toArray();
		for(int i=0;i<eigenFacesData.length;i++){
			double norm=Matrix2D.norm(eigenFacesData[i]);
			for(int j=0;j<eigenFacesData[i].length;j++){
				double v=eigenFacesData[i][j];
				eigenFacesData[i][j]=v/norm;
			}
		}
		return new Matrix2D(eigenFacesData);
	}
	
	public void sortEigenVectors(double[] eigenValues,double[][]eigenVectors){
		Hashtable<Double,double[]> table =  new Hashtable<Double,double[]> ();	
		Double[] evals=new Double[eigenValues.length];
		getEigenValuesAsDouble(eigenValues, evals);		
		fillHashtable(eigenValues, eigenVectors, table, evals);
		ArrayList<Double> keylist = sortKeysInReverse(table);				
		updateEigenVectors(eigenVectors, table, evals, keylist);		
		Double[] sortedkeys=new Double[keylist.size()];
		keylist.toArray(sortedkeys);//store the sorted list elements in an array
		//use the array to update the original double[]eigValues
		updateEigenValues(eigenValues, sortedkeys);		
	}
	
	private void getEigenValuesAsDouble(double[] eigenValue, Double[] evals) {
		for(int i=0;i<eigenValue.length;i++){
			evals[i]=new Double(eigenValue[i]);
		}
	}
	
	private ArrayList<Double> sortKeysInReverse(Hashtable<Double, double[]> table) {
		Enumeration<Double> keys=table.keys();
		ArrayList<Double> keylist=Collections.list(keys);		
		Collections.sort(keylist,Collections.reverseOrder());//largest first
		return keylist;
	}
	
	private void updateEigenValues(double[] eigenValue, Double[] sortedkeys) {
		for(int i=0;i<sortedkeys.length;i++){
			Double dbl=sortedkeys[i];
			double dblval=dbl.doubleValue();
			eigenValue[i]=dblval;
		}
	}
	
	private void updateEigenVectors(double[][] eigenVector,
			Hashtable<Double, double[]> table, Double[] evals,
			ArrayList<Double> keylist) {
		for(int i=0;i<evals.length;i++){
			double[] ret=table.get(keylist.get(i));
			setColumn(eigenVector,ret,i);
		}
	}
	
	private void fillHashtable(double[] eigenValues, double[][] eigenVectors,
			Hashtable<Double, double[]> table, Double[] evals) {
		for(int i=0;i<eigenValues.length;i++){
			Double key=evals[i];
			double[] value=getColumn(eigenVectors ,i);			
			table.put(key,value);
		}
	}
	
	private double[] getColumn( double[][] mat, int j ){
		int m = mat.length;
		double[] res = new double[m];
		for ( int i = 0; i < m; ++i ){
			res[i] = mat[i][j];
		}
		return(res);
	}
	
	private void setColumn(double[][] mat,double[] col,int c){
		int len=col.length;
		for(int row=0;row<len;row++){
			mat[row][c]=col[row];
		}
	}
	
	private Matrix2D getNormalisedImagesData(BufferedImage[] bufImgs){
		int imageWidth=bufImgs[0].getWidth();
		int imageHeight=bufImgs[0].getHeight();
		int rows=bufImgs.length;
		int cols=imageWidth * imageHeight;
		double[][] data=new double[rows][cols];
		for(int i=0;i<rows;i++){						
			bufImgs[i].getData().getPixels(0,0, imageWidth,imageHeight,data[i]);
		}
		Matrix2D imagesData=new Matrix2D(data);
		imagesData.normalise();
		return imagesData;		
	}
	
	private void checkImageDimensions(List<String> filenames,
			BufferedImage[] bufimgs) throws FaceRecError {
		int imgheight=0;
		int imgwidth=0;
		for(int i=0;i<bufimgs.length;i++){			
			if(i == 0){
				imgheight = bufimgs[i].getHeight();
				imgwidth=bufimgs[i].getWidth();				
			}
			if((imgheight != bufimgs[i].getHeight()) || (imgwidth != bufimgs[i].getWidth())){
				String response = "all images should have same dimensions! " + filenames.get(i) + " is of diff size";
	            throw new FaceRecError(response);
			}
		}
	}
	
	public BufferedImage[] getGrayScaleImages(List<String> filenames) throws FaceRecError{
		BufferedImage b = null;
		BufferedImage[] bufimgs = new BufferedImage[filenames.size()];
		Iterator<String> it = filenames.iterator();		
		int i=0;
		while(it.hasNext()){
			String fn = it.next();		
			File f = new File(fn);
			if (f.isFile()){
				try{
					b = ImageIO.read(new File(fn));
				}catch(IOException ioe){
					throw new FaceRecError(ioe.getMessage());
				}
				if(b != null){
					b = convertToGray(b);								
					bufimgs[i++] = b;
				}
			}
		}
		return bufimgs;			
	}
	
	private BufferedImage convertToGray(BufferedImage img) throws FaceRecError{
		BufferedImage gray = null;
		try{
		
			gray = new BufferedImage(img.getWidth(),img.getHeight(),
		              BufferedImage.TYPE_BYTE_GRAY);
			ColorConvertOp op = new ColorConvertOp(
		              img.getColorModel().getColorSpace(),
		              gray.getColorModel().getColorSpace(),null);
		    op.filter(img,gray); 
			return gray;
		}catch(Exception e){
			throw new FaceRecError("grayscale conversion failed:\n" + e.getMessage());
		}		
	}
	
	private void writeCache(String dir,FaceBundle cachedata) throws IOException {		
			FileOutputStream fout = null;
			ObjectOutputStream fos = null;
				fout = new FileOutputStream(dir + File.separator+"mycache.cache");
			    fos = new ObjectOutputStream(fout);
			    fos.writeObject(cachedata);
			    debug("wrote cache");
			    //fos.close();
			    fout.close();	
	}
	
	private FaceBundle getOldFacesBundle(String dir)throws IOException, ClassNotFoundException {
		FileInputStream fin = new FileInputStream(dir+File.separator+"mycache.cache");
		ObjectInputStream fo = new ObjectInputStream(fin);
		FaceBundle oldBundle = (FaceBundle)fo.readObject();
		fo.close();
		//fin.close();
		return oldBundle;
	}
	
	private void validateSelectedEigenFacesNumber(int selectedNumOfEigenFaces,
			List<String> newFileNames) throws FaceRecError {
		int numImgs = newFileNames.size();
		if(selectedNumOfEigenFaces <= 0 || selectedNumOfEigenFaces >= numImgs){
			throw new FaceRecError("incorrect number of selectedeigenfaces used..\n use a number between 0 and upto "+numImgs);
		}
	}
	
	private List<String> getFileNames(String dir, String[] children) {
		java.util.List<String> imageFileNames = new java.util.ArrayList<String>();
		for (String i : children){			
			String fileName = dir + File.separator +i ;
			imageFileNames.add(fileName);
		}
		Collections.sort(imageFileNames);
		return imageFileNames;
	}
	
	public List<String> parseDirectory(String directoryName,String extension) throws FaceRecError{		
		final String ext = "." + extension;
		String[] children = null;
		File directory = new File(directoryName);
		
		if(directory.isDirectory()){
			children = directory.list(new FilenameFilter(){
				public boolean accept(File f,String name){
					return name.endsWith(ext);						
				}					
			});			
		}else{
			throw new FaceRecError(directoryName + " is not a directory");
		}		
		return getFileNames(directoryName, children);
	}
	
	public void checkCache(String dir,String extension,int selectedNumOfEigenFaces) throws FaceRecError,IOException{
		List<String> newFileNames = parseDirectory(dir,extension);
		FaceBundle oldBundle = null;
		try{
			validateSelectedEigenFacesNumber(selectedNumOfEigenFaces, newFileNames);
			oldBundle = getOldFacesBundle(dir);
			processCache(dir,newFileNames,oldBundle,selectedNumOfEigenFaces);
		}
		catch(FileNotFoundException e){
			debug("cache file not found");
			doCalculations(dir,newFileNames,selectedNumOfEigenFaces);
		}
		catch(Exception e){
			throw new FaceRecError(e.getMessage());
		}
	}
	
	private void processCache(String dir,List<String>newFileNames,FaceBundle oldBundle,int selectedNumOfEigenFaces) throws FaceRecError, IOException{
		List<String>oldFileNames = oldBundle.getImageFileNamesList();
		if(newFileNames.equals(oldFileNames)){
			this.bundle = oldBundle;			
			this.weights = calculateWeights(oldBundle,selectedNumOfEigenFaces);
		}
		else{
			debug("folder contents changed");
			doCalculations(dir,newFileNames,selectedNumOfEigenFaces);
		}
	}
	
	private void createImageFromArray(String filename,double[] imgdata,int wd) throws IOException{
		BufferedImage bufimg = new BufferedImage(wd,imgdata.length/wd, BufferedImage.TYPE_BYTE_GRAY);
		Raster rast = bufimg.getData();			
		WritableRaster wr = rast.createCompatibleWritableRaster();		
		double maxValue = Double.MIN_VALUE;
		double minValue = Double.MAX_VALUE;		
		for(int i = 0; i<imgdata.length; i++){
			maxValue = Math.max(maxValue, imgdata[i]);
			minValue = Math.min(minValue, imgdata[i]);			
		}
		for(int j=0;j<imgdata.length;j++){
			imgdata[j] = ((imgdata[j]-minValue)*255)/(maxValue-minValue);
		}		
		wr.setPixels(0, 0, wd,imgdata.length/wd, imgdata);
		bufimg.setData(wr);
		File newfile = new File(filename);
		ImageIO.write(bufimg, "png", newfile);		
	
	}
	
	private void makeNewFolder(String fldr){
		File folder = new File(fldr);
		if (folder.isDirectory()){
			printError("folder:"+fldr+" exists");
			deleteContents(folder);
		}
		else{
			printError("no such folder as:"+fldr);
			boolean madeFolder = folder.mkdir();
			if (! madeFolder){
				printError("could not create folder :"+fldr);
				
			}
		}
	}
	
	private void deleteContents(File f) {
		File[] files = f.listFiles();
		for(File i:files){
			delete(i);
		}
	}
	
	private void delete(File i) {
		if(i.isFile()){
			boolean deleted = i.delete();
			if (!deleted){
				printError("file:"+i.getPath() +"could not be deleted");
			}
		}
	}
	
	public void reconstructFaces(int selectedeigenfaces) throws IOException{		
		double[][]eigenfacesArray = bundle.getEigenFaces();		
		Matrix2D egnfacesMatrix = new Matrix2D(eigenfacesArray);
		Matrix2D egnfacesSubMatrix = egnfacesMatrix.getSubMatrix(selectedeigenfaces);
		double[] eigenvalues = bundle.getEigenValues();
		Matrix2D eigenvalsMatrix = new Matrix2D(eigenvalues,1);
		Matrix2D eigenvalsSubMatrix = eigenvalsMatrix.transpose().getSubMatrix(selectedeigenfaces);
		
		/*
		 * the term 'phi' is used to denote mean subtracted reconstructed imagedata  
		 * since that term appears in T&P doc.The term 'xnew' denotes phi+average_image
		 *
		 **/		
		double[][] phi = getPhiData(egnfacesSubMatrix, eigenvalsSubMatrix);		
		double[][] xnew = addAverageFaceData(phi);		
		String reconFolderName = ".."+ File.separator+"reconfaces";
		String ext = ".png";		
		reconstructPhiImages(phi, reconFolderName, ext);		
		reconstructOriginalImages(xnew, reconFolderName, ext);
		debug("reconstruction over");		
	}
	
	private double[][] getPhiData(Matrix2D egnfacesSubMatrix,Matrix2D eigenvalsSubMatrix) {
		double[]evalsub = eigenvalsSubMatrix.flatten();		
		Matrix2D tempEvalsMat = new Matrix2D(weights.length,evalsub.length);
		tempEvalsMat.replaceRowsWithArray(evalsub);
		Matrix2D tempmat = new Matrix2D(weights);
		tempmat.multiplyElementWise(tempEvalsMat);				
		Matrix2D phinewmat = tempmat.multiply(egnfacesSubMatrix);		
		double[][]phi = phinewmat.toArray();
		return phi;
	}
	
	private double[][] addAverageFaceData(double[][] phi) {
		double[][] xnew = new double[phi.length][phi[0].length];
		double[] avgface = bundle.getAvgFace();
		for(int i = 0;i<phi.length;i++){
			for(int j = 0; j < phi[i].length; j++){
				xnew[i][j] = phi[i][j] + avgface[j];
			}
		}
		return xnew;
	}
	
	private void reconstructOriginalImages(double[][] xnew,String reconFolderName, String ext) throws IOException {
		String prefix;
		prefix = "xnew";
		int imgwidth = bundle.getImageWidth();
		for(int i = 0; i < xnew.length; i++){
			double[] xnewdata = xnew[i];
			String filename = reconFolderName + File.separator + prefix + i + ext;			
			createImageFromArray(filename,xnewdata,imgwidth);
		}
	}
	
	private void reconstructPhiImages(double[][] phi, String reconFolderName,String ext) throws IOException {
		int imgwidth = bundle.getImageWidth();		
		makeNewFolder(reconFolderName);
		String prefix = "phi";		
		for(int i = 0; i < phi.length; i++){
			double[] phidata = phi[i];
			String filename = reconFolderName + File.separator + prefix + i + ext;			
			createImageFromArray(filename,phidata,imgwidth);
		}		
	}
	
	private int getNumofFacesVal(String numofFaces)throws NumberFormatException{
		return new Integer(numofFaces).intValue();
	}
	
	private double getThresholdVal(String threshold)throws NumberFormatException{
		return new Double(threshold).doubleValue();
	}
	
	private void validateSelectedImageFileName(String faceimagename) throws FaceRecError{
		if ( (faceimagename.length() == 0) || (!isImage(faceimagename)) ){
			throw new FaceRecError("select an imagefile");
		}
	}
	
	private boolean isImage(String imagefilename){
		boolean isimage = false;
		try{
            	BufferedImage bi = ImageIO.read(new File(imagefilename));
	            if (bi != null){
	            	isimage = true;
	            }
		    }catch(Exception e){
		    	isimage = false;
		    }
		return isimage; 
	}
	
	private void validateSelectedFolderName(String foldername) throws FaceRecError{
		if (foldername.length() == 0){
			throw new FaceRecError("select a folder");
		}
	}
	
	public MatchResult processSelections(String faceImageName,String directory,String numofFaces,String threshold) {
		MatchResult result = null;
		int numFaces = 0;
		double thresholdVal = 0.0;
		try{
			validateSelectedImageFileName(faceImageName);
			validateSelectedFolderName(directory);
			numFaces = getNumofFacesVal(numofFaces);
			thresholdVal = getThresholdVal(threshold);
			String extension = getFileExtension(faceImageName);
			checkCache(directory,extension,numFaces);
			reconstructFaces(numFaces);
			result = findMatchResult(faceImageName,numFaces,thresholdVal);
		}catch(Exception e){
			result = new MatchResult(false,null,Double.NaN,e.getMessage());
		}
		return result;
	}
	
	//demo main
	public static void main(String[] args){
		long start = System.currentTimeMillis();
        if (args.length< 4){
        	printError("Usage:  java FaceRec  imageName imageDir  numberOfEigenfaces threshold");
        }
		String imgToCheck = args[0];
		String imgDir = args[1];
		String numFaces = args[2];
		String thresholdVal = args[3];
		MatchResult r = new FaceRec().processSelections(imgToCheck,imgDir,numFaces,thresholdVal);
		if (r.getMatchSuccess()){
			debug(imgToCheck + " matches "+r.getMatchFileName()+" at distance=" + r.getMatchDistance());
			
		}else{
			printError("match failed:" + r.getMatchMessage());
		}
		long end = System.currentTimeMillis();
		debug("time taken ="+ (end - start) / 1000.0 +" seconds");
	}
	
	public static String getFileExtension(String filename) {
	    String ext = "";
	    int i = filename.lastIndexOf('.');
	    if (i > 0 &&  i < filename.length() - 1) {
	        ext = filename.substring(i+1).toLowerCase();
	    }
	    return ext;
	} 
	public static void printError(String msg) {
		System.err.println(msg);		
	}
	
	public static void debug(String msg) {
		System.out.println(msg);		
	}
}
