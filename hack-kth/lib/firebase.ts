import { initializeApp } from 'firebase/app';
import { getFirestore } from 'firebase/firestore';

const firebaseConfig = {
  apiKey: "AIzaSyCBfMEyPgV90DOYfee00QyE-RtjlTuGdCE",
  authDomain: "kthackproject.firebaseapp.com",
  projectId: "kthackproject",
  storageBucket: "kthackproject.firebasestorage.app",
  messagingSenderId: "677082017128",
  appId: "1:677082017128:web:8baacfca2b68583e6d7add",
  measurementId: "G-8017LYNZMY",
};

const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);
export default app; 