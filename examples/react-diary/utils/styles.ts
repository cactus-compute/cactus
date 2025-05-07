import { StyleSheet } from "react-native";

export const styles = StyleSheet.create({
    container: {
      flex: 1,
      padding: 20,
    },
    input: {
      borderWidth: 1,
      borderColor: '#ccc',
      padding: 10,
      marginBottom: 10,
      borderRadius: 5,
      minHeight: 100, 
      textAlignVertical: 'top',
    },
    entryContainer: {
      padding: 10,
      borderBottomWidth: 1,
      borderBottomColor: '#eee',
    },
    entryDate: {
      fontSize: 12,
      color: '#888',
      marginBottom: 5,
    },
    entryTitle: {
      fontWeight: 'bold',
      fontSize: 16,
      marginBottom: 5,
    },
  });
  