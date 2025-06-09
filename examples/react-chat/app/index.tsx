import { useState, useEffect } from 'react';
import { KeyboardAvoidingView, Platform, ScrollView, View, Text } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Header from '@/components/Header';
import { stopWords } from '@/utils/constants';
import { initLlamaContext } from '@/utils/modelUtils';
import { LlamaContext } from 'cactus-react-native-2';
import { Message, MessageBubble } from '@/components/Message';
import { MessageField } from '@/components/MessageField';
let context: LlamaContext | null = null;

export default function HomeScreen() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [message, setMessage] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0.0);
  
  const handleSendMessage = async () => {
    setIsGenerating(true);
    const updatedMessages: Message[] = [...messages, { role: 'user', content: message }];
    setMessages(updatedMessages);
    setMessage('');
    await getLLMcompletion(updatedMessages);
  }

  useEffect(() => {
    const initializeContext = async () => {
      context = await initLlamaContext((progress) => {
        setDownloadProgress(progress);
      });
      if (context) {
        setIsModelLoaded(true);
      }
    };
    initializeContext();
  }, []);

  const getLLMcompletion = async (messages: Message[]) => {
    if (!context) {
      console.error('Model not yet loaded');
      return;
    }
    let _llmResponse = ''
    setMessages(prev => [...prev, { role: 'assistant', content: _llmResponse }]);
    await context.completion(
      {
        messages: messages,
        n_predict: 512,
        stop: stopWords,
      },
      (data: any) => { // streaming partial completion callback
        if (data.token) {
          _llmResponse += data.token;
          setMessages(prev => [
            ...prev.slice(0, prev.length - 1), 
            { role: 'assistant', content: prev[prev.length - 1].content + data.token }]
          );
        }
      }
    );

    setIsGenerating(false);
  } 

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: '#CCCCCC' }}>
        <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={{ flex: 1 }}>
        <Header />
        <ScrollView style={{ flex: 1, backgroundColor: '#FFFFFF', width: '100%', padding: '2%'}}>
          {
            isModelLoaded ? (
              messages.map((message, index) => <MessageBubble message={message} key={index} />)
            ) : (
              <View style={{ alignItems: 'center' }}>
                <Text>Hold tight... downloading model ({downloadProgress}%)</Text>
              </View>
            )
          }
        </ScrollView>
        <MessageField 
          message={message} 
          setMessage={setMessage} 
          handleSendMessage={handleSendMessage} 
          isGenerating={isGenerating} 
        />
        </KeyboardAvoidingView>
    </SafeAreaView>
  );
}
