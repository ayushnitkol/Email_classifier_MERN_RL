import React from 'react';
import { MdCropSquare } from 'react-icons/md';
import { MdOutlineStarBorder } from "react-icons/md";
import { useDispatch } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import { setSelectedEmail } from '../redux/appSlice';

const formatDate = (iso) => {
  if(!iso) return "";
  const d = new Date(iso);
  // show date or time nicely
  const diff = (Date.now() - d.getTime()) / (1000 * 60 * 60 * 24);
  if (diff < 1) return d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
  if (diff < 7) return d.toLocaleDateString([], { weekday: 'short' });
  return d.toLocaleDateString();
};

const Email = ({ email }) => {
  const navigate = useNavigate();
  const dispatch = useDispatch();

  const openMail = () => {
    dispatch(setSelectedEmail(email));
    navigate(`/mail/${email._id}`);
  };

  return (
    <div
      onClick={openMail}
      className="flex items-start gap-3 px-4 py-3 rounded-lg hover:bg-gray-50 cursor-pointer transition border-b"
    >
      <div className="w-10 h-10 rounded-full bg-indigo-50 text-indigo-700 flex items-center justify-center font-semibold">
        {email?.from?.[0] || 'U'}
      </div>

      <div className="flex-1">
        <div className="flex justify-between items-start">
          <div>
            <h3 className="font-medium text-gray-800">{email?.subject || "No subject"}</h3>
            <p className="text-sm text-gray-500 line-clamp-1 mt-1">{email?.message}</p>
          </div>

          <div className="text-right">
            <div className="text-sm text-gray-400">{formatDate(email?.createdAt)}</div>
            <div className="mt-2 text-gray-400">{email?.to}</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Email;
