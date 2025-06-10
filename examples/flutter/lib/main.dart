import 'package:flutter/material.dart';
import 'chat_screen.dart'; // Import the new chat screen

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cactus Flutter Chat',
      theme: ThemeData(
        primarySwatch: Colors.teal, // A slightly different theme for the example
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: const ChatScreen(), // Use ChatScreen as the home
    );
  }
}
