import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { DiaryEntry, downloadModelIfNotExists, getFullModelPath, getModelNameFromUrl } from '@/utils/functions';
import { styles } from '@/utils/styles';
import { LlamaContext, initLlama } from 'cactus-react-native';
import * as FileSystem from 'expo-file-system';
import React, { useEffect, useState } from 'react';
import { Button, FlatList, Platform, TextInput, View } from 'react-native';

export default function HomeScreen() {
  const [entry, setEntry] = useState('');
  const [entries, setEntries] = useState<DiaryEntry[]>([]);
  const [context, setContext] = useState<LlamaContext | null>(null);
  const [modelDownloaded, setModelDownloaded] = useState(false);
  
  const modelUrl = 'https://huggingface.co/Mungert/gemma-3-4b-it-qat-q4_0-GGUF/resolve/main/gemma-3-4b-it-qat-q4_0-q3_k_s.gguf'

  const animateTitle = (entryId: string, title: string) => {
    let charCount = 0;
    const intervalId = setInterval(() => {
      charCount++;
      setEntries(prevEntries => 
        prevEntries.map(item => 
          item.id === entryId 
            ? { ...item, visibleChars: charCount } 
            : item
        )
      );
      
      if (charCount >= title.length) {
        clearInterval(intervalId);
      }
    }, 25);
  };

  const downloadModel = async () => {return await downloadModelIfNotExists(modelUrl)}

  const loadModel = async () => {
    const modelPath = getFullModelPath(getModelNameFromUrl(modelUrl));
    if ((await FileSystem.getInfoAsync(modelPath)).exists) {
      const context = await initLlama({
      model: modelPath,
      use_mlock: true,
      n_ctx: 2048,
        n_gpu_layers: Platform.OS === 'ios' ? 99 : 0
      });
      setContext(context);
    }
  }

  useEffect(() => {
    downloadModel().then((result) => {
      setModelDownloaded(result);
    })
  }, []);

  useEffect(() => {
    if (modelDownloaded) {
      loadModel()
    }
  }, [modelDownloaded])

  const addEntry = async () => {
    if (entry.trim() === '') return;
    const newEntryId = Date.now().toString();
    setEntries([...entries, { 
      id: newEntryId, 
      text: entry, 
      date: new Date(), 
      title: '', 
      visibleChars: 0
    }]);
    if (context) {
      const fullPrompt = `You are tasked with creating a four-word title for a diary entry. The title should be a single sentence that captures the essence of the entry.

      Here is the entry:
      ${entry}

      Here is a very very very concise, four-word title (asbolutely stick to four words, do not use symbols or punctuation):
      ` 
      const result = await context.completion({
        prompt: fullPrompt,
        temperature: 0.7,
        n_predict: 8,
      });
      const generatedTitle = result.content.trim().replace(/[^\w\s]/g, '');
      
      setEntries(prevEntries => {
        return prevEntries.map(item => 
          item.id === newEntryId ? { ...item, title: generatedTitle, visibleChars: 0 } : item
        );
      });
      
      animateTitle(newEntryId, generatedTitle);
    }
    setEntry('');
  };

  return (
    <ThemedView style={styles.container}>
      <TextInput
        style={styles.input}
        placeholder="Write your thoughts..."
        value={entry}
        onChangeText={setEntry}
        multiline
      />
      <Button title="Add Entry" onPress={addEntry} disabled={!modelDownloaded}/>
      <FlatList
        data={entries}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <View style={styles.entryContainer}>
            <ThemedText style={styles.entryDate}>{item.date.toLocaleDateString()}</ThemedText>
            <ThemedText style={styles.entryTitle}>
              {item.visibleChars !== undefined
                ? item.title.substring(0, item.visibleChars)
                : item.title}
            </ThemedText>
            <ThemedText>{item.text.length > 45 ? `${item.text.slice(0, 45)}...` : item.text}</ThemedText>
          </View>
        )}
      />
    </ThemedView>
  );
}