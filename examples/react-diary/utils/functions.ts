import * as FileSystem from 'expo-file-system';
import { Platform } from 'react-native';


export const getModelDirectory = () => 
    Platform.OS === 'ios' 
      ? `${FileSystem.documentDirectory}local-models/`
      : `${FileSystem.cacheDirectory}local-models/`;
  
  export const getFullModelPath = (fileName: string) => 
    `${getModelDirectory()}${fileName}`;
  
  export const getModelNameFromUrl = (modelUrl: string) => 
    modelUrl.split('/').pop() || 'model.gguf'
  
  export const downloadModelIfNotExists = async (modelUrl: string): Promise<boolean> => {
    const modelName = getModelNameFromUrl(modelUrl)
    const fullModelPath = getFullModelPath(modelName);
    console.log('fullModelPath', fullModelPath)
    if (!(await FileSystem.getInfoAsync(getModelDirectory())).exists) {
      console.log('Model directory does not exist');
      await FileSystem.makeDirectoryAsync(getModelDirectory(), {
        intermediates: true
      });
      console.log('Created model directory');
    } else {
      console.log('Model directory exists');
    }
    if ((await FileSystem.getInfoAsync(fullModelPath)).exists) {
      console.log('Model file exists');
      return true;
    }
    console.log('downloading model...')
    const downloadResult = await FileSystem.downloadAsync(modelUrl, fullModelPath)
    console.log('downloadResult', downloadResult)
    return true;
  }
  
  export interface DiaryEntry {
    id: string;
    text: string;
    date: Date;
    title: string;
    visibleChars?: number;
  }